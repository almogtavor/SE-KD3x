# Selective LM Head for Memory-Efficient Token-Selective KD

## Overview

Three config flags control the selective lm_head flow:

| Flag | Effect |
|------|--------|
| `teacher_selective_lm_head` | Teacher lm_head on N_sel positions only (avoids `[B,L,V_teacher]`) |
| `student_selective_lm_head` | Student lm_head with grad on N_sel positions only (avoids `[B,L,V_student]` forward+backward) |
| `selective_lm_head_same_flow` | Force the selective flow even when both selective flags are false (useful for same-flow baselines) |

**Important**: This flow is **incompatible with `offline_cache=True`** — it requires an online teacher forward pass. Set `NO_OFFLINE=1` when using selective lm_head.

## Flow (when enabled)

1. **Student base transformer** (`student.model`) runs with grad on all positions → `[B, T, D_student]` hidden states.
2. **Student entropy** is computed via **chunked streaming** (default `chunk_size=128`) under `torch.no_grad()`. Only scalar entropy per position is stored; logits are discarded per chunk.
3. **Top-k% positions** are selected by highest student entropy within valid next-token positions.
4. **Student lm_head**:
   - If `student_selective_lm_head=True`: runs *with grad* only on selected positions → `[N_sel, V_student]`
   - Otherwise: runs on all positions, then indexed to selected → `[N_sel, V_student]`
5. **Teacher forward**:
   - If `teacher_selective_lm_head=True`: base transformer runs on all positions, lm_head only on selected → `[N_sel, V_teacher]`
   - Otherwise: full forward, then indexed to selected → `[N_sel, V_teacher]`
6. **KD loss** is computed on the N_sel positions using `F.kl_div(log_target=True)` for memory efficiency.
7. **CE loss** (if `alpha_ce > 0`):
   - If `enable_ce_on_all_tokens=True`: requires full student logits (computed separately if using selective student)
   - Otherwise: CE computed on selected positions only

Each flag can be enabled independently to isolate its memory contribution.

## Behavior Notes

- **Chunked streaming entropy**: The lm_head is applied to `chunk_size` positions at a time (configurable via `entropy_streaming_chunk_size`, default 128) under `torch.no_grad()`. Peak memory for entropy: `[B, chunk_size, V]` (~78 MB for chunk_size=128, V=152k) instead of `[B, L, V]` (~580 MB).
- **Student selective lm_head**: Reduces backward memory because gradients flow only through `[N_sel, V]` instead of `[B, L, V]`. Hidden states are freed early if CE loss doesn't need them.
- **Teacher selective lm_head**: Avoids full `[B,L,V_teacher]` logits by running base transformer first, then lm_head only on selected hidden states. This reduces logits memory and lm_head compute, but does not reduce teacher base transformer compute.
- **Same-flow baseline**: Use `selective_lm_head_same_flow=True` with both selective flags false to run the identical code path without actually enabling selective lm_head (for fair timing comparisons).

## Memory Savings (Measured)

Qwen3-8B teacher → Qwen3-1.7B student, B=2, T=2048, RTX 3090 (24GB each):

| Configuration | Teacher Peak | Student Peak | Speed |
|---------------|-------------|--------------|-------|
| Full KD (k=100%) | 16.7 GB | 10.6 GB | baseline |
| Selective (k=20%) | 16.7 GB | 7.8 GB | ~26% faster |

Student savings at k=20%: **2.8 GB (−26%)** — transient lm_head allocations eliminated.

## Configuration

```bash
# Both selective (max memory savings)
TEACHER_SELECTIVE_LM_HEAD=1 \
STUDENT_SELECTIVE_LM_HEAD=1 \
TOPK_TOK_SELECTION_METRIC=student_entropy \
NORMALIZE_TOPK_BY_LENGTH=1 \
K_PERCENT=20 \
ALPHA_CE=0.0 \
NO_OFFLINE=1 \
LOG_PEAK_MEMORY=1 \
  sbatch train.slurm top-k-tok 20 light mytag

# Same-flow baseline (identical code path, no actual selection savings)
SELECTIVE_LM_HEAD_SAME_FLOW=1 \
K_PERCENT=100 \
NO_OFFLINE=1 \
  sbatch train.slurm top-k-tok 100 light baseline
```

### Environment Variables

| Variable | Config Field | Description |
|----------|-------------|-------------|
| `TEACHER_SELECTIVE_LM_HEAD` | `teacher_selective_lm_head` | Enable teacher-selective lm_head |
| `STUDENT_SELECTIVE_LM_HEAD` | `student_selective_lm_head` | Enable student-selective lm_head |
| `SELECTIVE_LM_HEAD_SAME_FLOW` | `selective_lm_head_same_flow` | Force selective flow without actual savings |
| `ENTROPY_STREAMING_CHUNK_SIZE` | `entropy_streaming_chunk_size` | Chunk size for streaming entropy (default 128) |
| `LOG_PEAK_MEMORY` | `log_peak_memory` | Log peak GPU memory per step |
| `NO_OFFLINE` | Sets `offline_cache=False` | Required for selective lm_head |

## Files

| File | Description |
|------|-------------|
| `selective_lm_head.py` | Core functions: streaming entropy, selective logits, teacher forward |
| `forward_batch_runner.py` | `_run_selective_lm_head_flow()` branch in training loop |
| `config.py` | Config fields: `teacher_selective_lm_head`, `student_selective_lm_head`, etc. |
| `trainer.py` | Peak memory tracking per step |

### Key Functions in `selective_lm_head.py`

- `_streaming_entropy_from_hidden_states()` — Chunked entropy computation
- `compute_student_entropy_and_select()` — Select top-k% by student entropy
- `selective_student_logits()` — Apply lm_head with grad on selected positions
- `selective_teacher_logits()` — Run teacher base + selective lm_head

## Measuring Memory

Set `LOG_PEAK_MEMORY=1` to log `torch.cuda.max_memory_allocated()` per step.

Results appear in:
- **W&B**: `train/peak_memory_gb`
- **Efficiency CSV**: `peak_memory_gb` column (average over all steps)

For detailed memory timelines, enable the PyTorch profiler:
```bash
PROFILER_ENABLED=1 PROFILER_ACTIVE=10 python run_distillation.py ...
```
Then use `tools/plot_memory_comparison.py` to generate side-by-side comparison PDFs from the resulting trace JSONs.

## Speed Logging

When the selective lm_head flow is active, per-step timing metrics are logged to W&B:

| Metric | Description |
|--------|-------------|
| `train/selective_lm_head_n_selected` | Number of positions selected |
| `train/selective_lm_head_student_base_time_s` | Time for student base transformer |
| `train/selective_lm_head_entropy_time_s` | Time for streaming entropy + selection |
| `train/selective_lm_head_student_logits_time_s` | Time for student lm_head |
| `train/selective_lm_head_teacher_logits_time_s` | Time for teacher forward + lm_head |
| `train/selective_lm_head_total_time_s` | Total selective flow time |
| `train/selective_lm_head_student_enabled` | 1.0 if student-selective enabled |
| `train/selective_lm_head_teacher_enabled` | 1.0 if teacher-selective enabled |

These are CPU wall-clock timings (no explicit CUDA sync), so use them for relative comparisons across runs rather than absolute GPU time.

## Implementation Details

### KD Loss Computation

The selective flow uses `F.kl_div(log_target=True)` to avoid materializing intermediate probability tensors:

```python
s_log_probs = torch.log_softmax(s_logits_for_kd.float() / T, dim=-1)
t_log_probs = torch.log_softmax(t_logits_for_kd.float() / T, dim=-1)
kd_loss = F.kl_div(s_log_probs, t_log_probs, log_target=True, reduction='batchmean') * T2
```

### Why Student Peak is Same Regardless of `sel_teacher`

When `student_selective_lm_head=True`, PyTorch's autograd only tracks the indexed slice `s_logits_for_kd` (shape `[N_sel, V]`). Even if the teacher is run non-selectively (full `[B,L,V_teacher]`), this doesn't affect the student's backward pass or memory usage — the teacher logits are on a different device and used only under `torch.no_grad()`.

### Contiguous Selection Optimization

`selective_student_logits()` calls `.contiguous()` on the gathered hidden states before applying lm_head. This allows PyTorch to free the original `[B,L,D]` hidden states tensor earlier if no other references exist.

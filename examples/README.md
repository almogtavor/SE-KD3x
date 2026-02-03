# Example Scripts

This directory contains scripts to reproduce the main experiments from the paper.

## Scripts

| Script | Method | Expected Results |
|--------|--------|------------------|
| `run_full_kd.sh` | Full KD baseline | 64.4% acc, 7.3 PPL |
| `run_sekd.sh` | SE-KD (position selection) | **64.8%** acc, **6.9** PPL |
| `run_sekd3x.sh` | SE-KD₃ₓ (3-axis selection) | 64.4% acc, 7.3 PPL + **70% faster** |

## Running

```bash
# Make scripts executable
chmod +x examples/*.sh

# Run SE-KD (recommended starting point)
./examples/run_sekd.sh
```

## Hardware Requirements

- 2× GPUs with ≥24GB VRAM (e.g., RTX 3090, A100)
- Teacher (Qwen3-8B) on one GPU, Student (Qwen3-1.7B) on another
- ~80M tokens from FineWeb-Edu (~4-8 hours on 2× RTX 3090)

## Custom Configurations

### Different position selection metrics

```bash
# Teacher entropy (baseline)
--topk_tok_selection_metric teacher_entropy

# KL divergence
--topk_tok_selection_metric kl

# CE ratio (best perplexity)
--topk_tok_selection_metric ce_ratio
```

### Different selection budgets

```bash
# More aggressive selection (top 10%)
--k_percent 10

# Conservative selection (top 50%)
--k_percent 50
```

### Memory-efficient mode

Enable selective LM heads for reduced peak memory:

```bash
--teacher_selective_lm_head \
--student_selective_lm_head
```

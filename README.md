<div align="center">

# SE-KD<sub>3X</sub>

### Rethinking Selective Knowledge Distillation

<img src="assets/sekd_overview.png" width="90%">

[![Paper](https://img.shields.io/badge/Paper-OpenReview-b31b1b.svg)](https://openreview.net/attachment?id=zRxYXSdQlT&name=pdf)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg)](https://pytorch.org/)

**Dense supervision is unnecessary.** Distill only the **top-20% highest-entropy tokens** and match or beat Full KD-with massive efficiency gains.

[Quick Start](#-quick-start) · [Results](#-key-results) · [Method](#-method) · [Citation](#-citation)

</div>

---

## 🎯 TL;DR

We introduce **SE-KD** (Student-Entropy-guided KD) and **SE-KD<sub>3X</sub>** (multi-axis selection) for efficient LLM distillation:

| What We Do | Why Would It Work? |
|------------|--------------|
| 📍 **Position Selection** - Distill only top-k% tokens by student entropy | High-entropy "fork" tokens carry most learning signal |
| 📊 **Class Sampling** - Sample vocabulary classes via RS-KD | Unbiased gradients matches Full KD in expectation. Leads to 99.96% storage reduction when combined with sample selection |
| 📁 **Sample Selection** - Keep top-l% samples by avg. student entropy | Focus compute on informative training examples |

---

## 📊 Key Results

Trained on **80M tokens** from FineWeb-Edu, evaluated zero-shot on 3 seeds:

| Method | Accuracy ↑ | IFEval ↑ | PPL ↓ | ECE ↓ |
|--------|:----------:|:--------:|:-----:|:-----:|
| Full KD | 64.4 | 20.5 | 7.3 | **27.3** |
| **SE-KD** | **64.8** | **21.4** | **6.9** | 27.6 |
| SE-KD<sub>3X</sub> | 64.4 | 20.7 | 7.3 | 27.9 |

### Efficiency Gains with SE-KD<sub>3X</sub>

| Metric | Improvement |
|--------|:-----------:|
| 🚀 Wall Time | **-70%** |
| 💾 Peak Memory | **-18%** |
| 📦 Storage | **-99.96%** |

---

## ⚡ Quick Start

```bash
# Install
pip install -e .

# SE-KD: Student entropy-guided position selection (top 20%)
python run_distillation.py \
    --teacher_model Qwen/Qwen3-8B \
    --student_model Qwen/Qwen3-1.7B \
    --datasets fineweb \
    --fineweb_tokens 80000000 \
    --distill_type top-k-tok \
    --k_percent 20 \
    --topk_tok_selection_metric student_entropy \
    --alpha_ce 0.0 \
    --output_dir results/sekd
```

<details>
<summary><b>Full KD Baseline</b></summary>

```bash
python run_distillation.py \
    --teacher_model Qwen/Qwen3-8B \
    --student_model Qwen/Qwen3-1.7B \
    --datasets fineweb \
    --distill_type vanilla \
    --output_dir results/full_kd
```
</details>

<details>
<summary><b>SE-KD<sub>3X</sub> (Full 3-axis selection)</b></summary>

```bash
python run_distillation.py \
    --teacher_model Qwen/Qwen3-8B \
    --student_model Qwen/Qwen3-1.7B \
    --datasets fineweb \
    --distill_type top-k-tok \
    --k_percent 20 \
    --topk_tok_selection_metric student_entropy \
    --skip_by_frozen_student \
    --l_percent_samples_to_keep 20 \
    --rs_vocab_samples 64 \
    --offline_cache \
    --output_dir results/sekd3x
```
</details>

---

## 🧠 Method

Standard knowledge distillation applies teacher supervision uniformly at every token position-but this is suboptimal. Not all positions benefit equally from distillation.

We use **student entropy** `H(q_t)` to identify where the student is most uncertain. These high-uncertainty positions are where the teacher's guidance provides the most value. By selecting only the **top-20% highest student-entropy positions** for KD supervision, we focus compute on tokens that actually need it-and often outperform full dense supervision.

<details>
<summary><b>Memory Optimizations</b></summary>

Two optimizations reduce the memory footprint (see [detailed docs](sekd/distill/SELECTIVE_LM_HEAD_README.md)):

- **Chunked Entropy Computation** - Computes per-position entropy without materializing the full `[B,L,V]` logit tensor. Hidden states are projected through the LM head in small chunks (gradients disabled), reduced to `O(BL)` entropy scalars, then discarded.

- **Selective LM Head** - Computes logits only at selected positions. Teacher logits shrink from `[B,L,V]` to `[N_select,V]`, and backpropagation runs through only `N_select` positions instead of all `B×L`.

Together at `k=20%`: **-28% student peak memory**, **-9% teacher peak memory**.

<p align="center">
  <img src="assets/memory_comparison.png" width="90%">
</p>

</details>

<details>
<summary><b>Position-Importance Metrics</b></summary>

We systematically compared 9 importance signals to determine which positions benefit most from distillation:

| Metric | What it measures | Best For |
|--------|------------------|----------|
| **Student Entropy** `H(q_t)` | Student's uncertainty at position t | Overall best signal |
| Reverse KL `KL(q∥p)` | How much student diverges from teacher | Strong alternative |
| CE Ratio `CE_s/CE_t` | Relative difficulty (student CE / teacher CE) | Best perplexity |
| Teacher Entropy `H(p_t)` | Teacher's uncertainty at position t | Baseline comparison |

Student entropy consistently identifies the most valuable positions-where the student needs guidance most.

</details>

<details>
<summary><b>Selection Policies</b></summary>

Given an importance metric, how should we convert scores into a selection decision?

| Policy | How it works |
|--------|--------------|
| **Top-k%** | Deterministically select the k% highest-scoring positions per sequence |
| GLS | Global-level selection with thresholds normalized across batches via FIFO queue |
| Curriculum | Start with easy positions, progressively expand to harder ones during training |
| Pos RS-KD | Stochastically sample positions proportional to importance weights |

Top-k% with student entropy is the most reliable strategy at k=20%.


An illustration of position selection (SE-KD) when combined with RS-KD (blue), compared to RS-KD alone (light blue, selection at every position), on one sample:
<p align="center">
  <img src="assets/selection_axes.png" width="55%">
</p>

</details>

---

<details>
<summary><b>To Reproduce Paper Results</b></summary>

```bash
# Shared hyperparameters (Table 8 in paper)
--epochs 1 --batch_size 2 --gradient_accumulation_steps 8
--lr 1e-5 --max_seq_len 512 --kd_temperature 1.0 --alpha_ce 0.0
--seed 1337  # + 1338, 1339 for error bars
```

See [examples/](examples/) for full reproduction scripts.

</details>

---

## 📖 Citation

```bibtex
@article{sekd2025,
  title={Rethinking Selective Knowledge Distillation},
  author={Anonymous},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## 📄 License

Apache 2.0 - see [LICENSE](LICENSE)

---

<div align="center">

# SE-KD<sub>3X</sub>

### Rethinking Selective Knowledge Distillation

<img src="assets/sekd_overview.png" width="90%">

[![Paper](https://img.shields.io/badge/Paper-OpenReview-b31b1b.svg)](https://openreview.net/attachment?id=zRxYXSdQlT&name=pdf)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg)](https://pytorch.org/)

**Dense supervision is unnecessary.** Distill only the **top-20% highest-entropy tokens** and match or beat Full KD—with massive efficiency gains.

[Quick Start](#-quick-start) · [Results](#-key-results) · [Method](#-method) · [Citation](#-citation)

</div>

---

## 🎯 TL;DR

We introduce **SE-KD** (Student-Entropy-guided KD) and **SE-KD<sub>3X</sub>** (multi-axis selection) for efficient LLM distillation:

| What We Do | Why It Works |
|------------|--------------|
| 📍 **Position Selection** - Distill only top-k% tokens by student entropy | High-entropy "fork" tokens carry most learning signal |
| 📊 **Class Sampling** - Sample vocabulary classes via RS-KD | Unbiased gradients with 99.96% storage reduction |
| 📁 **Sample Filtering** - Keep top-l% samples by avg. student entropy | Focus compute on informative training examples |

<p align="center">
  <img src="assets/selection_axes.png" width="55%">
</p>

---

## 📊 Key Results

Trained on **80M tokens** from FineWeb-Edu, evaluated zero-shot:

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
<summary><b>🔧 Full KD Baseline</b></summary>

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
<summary><b>🚀 SE-KD<sub>3X</sub> (Full 3-axis selection)</b></summary>

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

Not all tokens are equal for distillation. High-entropy "fork" positions—where the model is uncertain between multiple valid continuations—carry most of the learning signal. We select the **top-20% highest student-entropy positions** for KD supervision.

<details>
<summary><b>Memory Optimizations</b></summary>

Two optimizations reduce the memory footprint (see [detailed docs](sekd/distill/SELECTIVE_LM_HEAD_README.md)):

- **Chunked Entropy Computation** — Computes per-position entropy without materializing the full `[B,L,V]` logit tensor. Hidden states are projected through the LM head in small chunks (gradients disabled), reduced to `O(BL)` entropy scalars, then discarded.

- **Selective LM Head** — Computes logits only at selected positions. Teacher logits shrink from `[B,L,V]` to `[N_select,V]`, and backpropagation runs through only `N_select` positions instead of all `B×L`.

Together at `k=20%`: **-28% student peak memory**, **-9% teacher peak memory**.

<p align="center">
  <img src="assets/memory_comparison.png" width="90%">
</p>

</details>

<details>
<summary><b>Position-Importance Metrics</b></summary>

| Metric | Best For |
|--------|----------|
| **Student Entropy** `H(q_t)` | Overall best signal |
| Reverse KL `KL(q\|\|p)` | Strong alternative |
| CE Ratio `CE_s/CE_t` | Best perplexity |
| Teacher Entropy `H(p_t)` | Baseline comparison |

</details>

<details>
<summary><b>Selection Policies</b></summary>

| Policy | Description |
|--------|-------------|
| **Top-k%** | Hard selection of k% highest-scoring positions |
| GLS | Global-level selection with FIFO queue |
| Curriculum | Easy → hard progression over training |
| Pos RS-KD | Stochastic position sampling |

</details>

---

<details>
<summary><b>📁 Project Structure</b></summary>

```
SE-KD3X/
├── sekd/                    # Core package
│   ├── config.py           # Training configuration
│   ├── distill/            # Distillation trainers & losses
│   │   ├── trainer.py      # Main Distiller class
│   │   ├── selective_lm_head.py  # Memory-efficient logit computation
│   │   └── _mixins/        # Modular components (entropy, GLS, etc.)
│   ├── data/               # Dataset utilities
│   ├── models/             # Model loading
│   └── training/           # Distributed training, caching
├── run_distillation.py     # Main entry point
├── examples/               # Example scripts
└── tests/                  # Unit tests
```

</details>

<details>
<summary><b>🔬 Reproduce Paper Results</b></summary>

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
@inproceedings{sekd2025,
  title={Rethinking Selective Knowledge Distillation},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025},
  note={Under review}
}
```

---

## 📄 License

Apache 2.0 - see [LICENSE](LICENSE)

---

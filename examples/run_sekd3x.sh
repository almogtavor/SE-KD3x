#!/bin/bash
# SE-KD3X: Multi-Axis Selection (Position + Class + Sample)
# - Position: Top-20% by student entropy
# - Sample: Top-20% by avg student entropy (pre-filtered)
# - Class: RS-KD with 64 sampled vocabulary classes
#
# Efficiency gains:
# - 70% wall time reduction
# - 18% peak memory reduction
# - 99.96% storage reduction

python run_distillation.py \
    --teacher_model Qwen/Qwen3-8B \
    --student_model Qwen/Qwen3-1.7B \
    --datasets fineweb \
    --fineweb_tokens 80000000 \
    --distill_type top-k-tok \
    --k_percent 20 \
    --topk_tok_selection_metric student_entropy \
    --skip_by_frozen_student \
    --l_percent_samples_to_keep 20 \
    --skip_samples_strategy entropy \
    --rs_vocab_samples 64 \
    --offline_cache \
    --offline_cache_mode entropy \
    --epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr 1e-5 \
    --max_seq_len 512 \
    --kd_temperature 1.0 \
    --alpha_ce 0.0 \
    --seed 1337 \
    --output_dir results/sekd3x

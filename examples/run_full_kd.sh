#!/bin/bash
# Full KD Baseline - Distill all tokens with KL divergence
# Qwen3-8B (teacher) → Qwen3-1.7B (student)
# Expected: ~64.4% avg accuracy, 7.3 PPL

python run_distillation.py \
    --teacher_model Qwen/Qwen3-8B \
    --student_model Qwen/Qwen3-1.7B \
    --datasets fineweb \
    --fineweb_tokens 80000000 \
    --distill_type vanilla \
    --epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr 1e-5 \
    --max_seq_len 512 \
    --kd_temperature 1.0 \
    --alpha_ce 0.0 \
    --seed 1337 \
    --output_dir results/full_kd

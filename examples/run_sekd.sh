#!/bin/bash
# SE-KD: Student Entropy-guided Position Selection
# Distill only top-20% positions by student entropy
# Expected: ~64.8% avg accuracy, 6.9 PPL (better than Full KD!)

python run_distillation.py \
    --teacher_model Qwen/Qwen3-8B \
    --student_model Qwen/Qwen3-1.7B \
    --datasets fineweb \
    --fineweb_tokens 80000000 \
    --distill_type top-k-tok \
    --k_percent 20 \
    --topk_tok_selection_metric student_entropy \
    --epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr 1e-5 \
    --max_seq_len 512 \
    --kd_temperature 1.0 \
    --alpha_ce 0.0 \
    --seed 1337 \
    --output_dir results/sekd

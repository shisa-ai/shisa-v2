#!/bin/bash

# 8xMI300X Node
# accelerate launch --num_processes 8 unlikelihood_train.py \

# 1xMI300X
python unlikelihood_train.py \
  --model_name_or_path shisa-ai/170-qwen3-8b-v2.1-dpo-1.2e7 \
  --dataset_name shisa-ai/cltl_correction_qwen3-8b-merge-nuslerp-170-162-161 \
  --output_dir /data/outputs/170-qwen3-8b-v2.1-dpo-1.2e7-leakfix \
  --log_file /data/outputs/170-qwen3-8b-v2.1-dpo-1.2e7-leakfix/run.log \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --leak_tokens_column leak_tokens \
  --corrected_token_column corrected_token \
  --epochs 5 \
  --alpha_ul 2.0 \
  --alpha_reinforce 0.1 \
  --verbose_diagnostics

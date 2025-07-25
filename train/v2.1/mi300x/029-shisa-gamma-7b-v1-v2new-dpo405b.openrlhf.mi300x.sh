#!/usr/bin/env bash
set -eo pipefail

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"           # keep your old project
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="028-shisa-gamma-7b-v1-v2new-sft"
DATA="shisa-ai/shisa-v2-dpo405b-shuffled"
OUT="029-shisa-gamma-7b-v1-v2new-dpo405b"

# GBS = 256 - LR = 5e-7 constant
# GBS = 128 - LR = 2.5e-7
# GBS = 64 - LR = 1.25e-7

deepspeed --module openrlhf.cli.train_dpo \
  --pretrain        "$MODEL" \
  --dataset         "$DATA" \
  --chosen_key      chosen \
  --rejected_key    rejected \
  --apply_chat_template \
  --beta            0.1 \
  --bf16 \
  --zero_stage      3 \
  --gradient_checkpointing \
  --flash_attn \
  --use_liger_kernel \
  --optimizer paged_adamw_8bit \
  --train_batch_size       64 \
  --micro_train_batch_size 8 \
  --max_len         2048 \
  --packing_samples \
  --max_epochs      1 \
  --learning_rate   1.25e-7 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_ratio 0.03 \
  --save_steps      -1 \
  --save_path       "$OUT" \
  --logging_steps   1 \
  --overlap_comm \
  --use_wandb       True \
  --wandb_org        augmxnt \
  --wandb_project    shisa-v2.1 \
  --wandb_run_name   "$OUT-mi300x"

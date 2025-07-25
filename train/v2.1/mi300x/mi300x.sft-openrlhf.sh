#!/usr/bin/env bash
set -eo pipefail

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"           # keep your old project
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_ASYNC_ERROR_HANDLING=1

MODEL="Qwen/Qwen3-30B-A3B"
# DATA="shisa-ai/shisa-v2-dpo-shuffled"
DATA="dpo.shisa-v2x.jsonl"
OUT="018-qwen-30ba3b-v2-dpo405b-clr"

# GBS = 256 - LR = 5e-7 constant
# GBS = 128 - LR = 2.5e-7
# GBS = 64 - LR = 1.25e-7

deepspeed --num_gpus 1 --module openrlhf.cli.train_dpo \
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
  --train_batch_size       64 \
  --micro_train_batch_size 2 \
  --max_len         2048 \
  --packing_samples \
  --max_epochs      1 \
  --learning_rate   2.5e-7 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_ratio 0.05 \
  --save_steps      -1 \
  --save_path       "$OUT" \
  --logging_steps   1 \
  --overlap_comm \
  --use_wandb       True \
  --wandb_org        augmxnt \
  --wandb_project    shisa-v2.1 \
  --wandb_run_name   "$OUT" 

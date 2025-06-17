#!/usr/bin/env bash
set -eo pipefail

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"           # keep your old project
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_ASYNC_ERROR_HANDLING=1

MODEL="shisa-ai/011-qwen3-8b-v2-sft"
DATA="shisa-ai/shisa-v2-dpo-shuffled"
OUT="/data/outputs/013-qwen3-8b-v2-dpo"

deepspeed --num_gpus 4 --module openrlhf.cli.train_dpo \
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
  --train_batch_size       256 \
  --micro_train_batch_size 8 \
  --max_len         2048 \
  --packing_samples \
  --max_epochs      1 \
  --learning_rate   5e-7 \
  --lr_warmup_ratio 0.05 \
  --save_steps      -1 \
  --save_path       "$OUT" \
  --logging_steps   1 \
  --overlap_comm \
  --use_wandb       013-qwen3-8b-v2-dpo

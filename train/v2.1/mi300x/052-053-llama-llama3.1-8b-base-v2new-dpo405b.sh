#!/usr/bin/env bash
set -eo pipefail

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"           # keep your old project
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_ASYNC_ERROR_HANDLING=1

# MODELS
ORIG_MODEL=${ORIG_MODEL:-"meta-llama/Llama-3.1-8B"}

SFT_CKPT=${SFT_CKPT:-"052-llama3.1-8b-base-v2new-sft"}
SFT_DATA="sft.shisa-v2-new.jsonl"
SFT_LR=${SFT_LR:-1e-5}
SFT_MBS=4

DPO_CKPT=${DPO_CKPT:-"053-llama3.1-8b-base-v2new-dpo405b"}
DPO_DATA="shisa-ai/shisa-v2-dpo405b-shuffled"
DPO_LR=${DPO_LR:-1.25e-7}
DPO_MBS=8

# SFT
deepspeed --num_gpus 1 --module openrlhf.cli.train_sft \
  --pretrain        "$ORIG_MODEL" \
  --dataset         "$SFT_DATA" \
  --input_key conversations \
  --apply_chat_template \
  --bf16 \
  --gradient_checkpointing \
  --flash_attn \
  --use_liger_kernel \
  --optimizer paged_adamw_8bit \
  --train_batch_size       128 \
  --micro_train_batch_size $SFT_MBS \
  --max_len 8192 \
  --packing_samples \
  --max_epochs      3 \
  --learning_rate   $SFT_LR \
  --lr_warmup_ratio 0.03 \
  --save_steps      90443 \
  --save_path       "$SFT_CKPT" \
  --logging_steps   1 \
  --eval_steps   -1 \
  --overlap_comm \
  --use_wandb       True \
  --wandb_org        augmxnt \
  --wandb_project    shisa-v2.1 \
  --wandb_run_name   "$SFT_CKPT-mi300x"

# DPO
deepspeed --module openrlhf.cli.train_dpo \
  --pretrain        "$SFT_CKPT" \
  --dataset         "$DPO_DATA" \
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
  --micro_train_batch_size $DPO_MBS \
  --max_len         2048 \
  --packing_samples \
  --max_epochs      1 \
  --learning_rate   $DPO_LR \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_ratio 0.03 \
  --save_steps      -1 \
  --save_path       "$DPO_CKPT" \
  --logging_steps   1 \
  --overlap_comm \
  --use_wandb       True \
  --wandb_org        augmxnt \
  --wandb_project    shisa-v2.1 \
  --wandb_run_name   "$DPO_CKPT-mi300x"

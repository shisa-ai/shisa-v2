#!/usr/bin/env bash
set -eo pipefail

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"           # keep your old project
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_ASYNC_ERROR_HANDLING=1

export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL="/data/outputs/shisa-v2-unphi4-14b-denoted-chotto-sft-3epoch"
DATA="chotto-dpo.jsonl"
OUT="/data/outputs/shisa-v2-unphi4-14b-denoted-chotto-sft-3epoch-dpo"

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
  --train_batch_size       128 \
  --micro_train_batch_size 8 \
  --max_len 4096 \
  --packing_samples \
  --max_epochs      1 \
  --learning_rate   3e-7 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_ratio 0.03 \
  --save_steps      -1 \
  --save_path       "$OUT" \
  --logging_steps   1 \
  --eval_steps   -1 \
  --overlap_comm \
  --use_wandb       True \
  --wandb_org        augmxnt \
  --wandb_project    shisa-v2.1 \
  --wandb_run_name   "$OUT"

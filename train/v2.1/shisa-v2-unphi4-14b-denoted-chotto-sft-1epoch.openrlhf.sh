#!/usr/bin/env bash
set -eo pipefail

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"           # keep your old project
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_ASYNC_ERROR_HANDLING=1

export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL="shisa-ai/shisa-v2-unphi4-14b-denoted"
DATA="chotto-sft.jsonl"
OUT="/data/outputs/shisa-v2-unphi4-14b-denoted-chotto-sft-1epoch"

deepspeed --module openrlhf.cli.train_sft \
  --pretrain        "$MODEL" \
  --dataset         "$DATA" \
  --input_key conversations \
  --apply_chat_template \
  --bf16 \
  --gradient_checkpointing \
  --flash_attn \
  --use_liger_kernel \
  --train_batch_size       128 \
  --micro_train_batch_size 8 \
  --max_len 8192 \
  --packing_samples \
  --max_epochs      1 \
  --learning_rate   7.5e-6 \
  --lr_warmup_ratio 0.05 \
  --save_steps      -1 \
  --save_path       "$OUT" \
  --logging_steps   1 \
  --eval_steps   -1 \
  --overlap_comm \
  --use_wandb       True \
  --wandb_org        augmxnt \
  --wandb_project    shisa-v2.1 \
  --wandb_run_name   "$OUT"

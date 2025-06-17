#!/usr/bin/env bash
set -eo pipefail

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"           # keep your old project
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_ASYNC_ERROR_HANDLING=1

export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
DATA="shisa-ai/shisa-v2new-sft-shuffled"
OUT="022-llama3.1-8b-v2new-sft"

# GBS = 256 - LR = 5e-7 constant
# GBS = 128 - LR = 2.5e-7
# GBS = 64 - LR = 1.25e-7

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
  --micro_train_batch_size 16 \
  --max_len 8192 \
  --packing_samples \
  --max_epochs      3 \
  --learning_rate   1e-5 \
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

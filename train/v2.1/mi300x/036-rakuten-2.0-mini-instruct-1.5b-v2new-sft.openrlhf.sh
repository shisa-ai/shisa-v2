#!/usr/bin/env bash

set -eo pipefail

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"           # keep your old project
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_ASYNC_ERROR_HANDLING=1

export CUDA_VISIBLE_DEVICES=0,1,2,3

# We replace the old tokenizer w/ a our known good shisa-v2-llama3.1-8b tokenizer...
MODEL="Rakuten/RakutenAI-2.0-mini-instruct"
DATA="sft.shisa-v2-new.jsonl"
OUT="036-rakuten-2.0-mini-instruct-1.5b-v2new-sft"

# GBS = 256 - LR = 5e-7 constant
# GBS = 128 - LR = 2.5e-7
# GBS = 64 - LR = 1.25e-7

deepspeed --num_gpus 1 --module openrlhf.cli.train_sft \
  --pretrain        "$MODEL" \
  --dataset         "$DATA" \
  --input_key conversations \
  --apply_chat_template \
  --bf16 \
  --gradient_checkpointing \
  --flash_attn \
  --use_liger_kernel \
  --optimizer paged_adamw_8bit \
  --train_batch_size       128 \
  --micro_train_batch_size 128 \
  --max_len 8192 \
  --packing_samples \
  --max_epochs      3 \
  --learning_rate   2.31e-5 \
  --lr_warmup_ratio 0.03 \
  --save_steps      90443 \
  --save_path       "$OUT" \
  --logging_steps   1 \
  --eval_steps   -1 \
  --overlap_comm \
  --use_wandb       True \
  --wandb_org        augmxnt \
  --wandb_project    shisa-v2.1 \
  --wandb_run_name   "$OUT-mi300x"

#  --aux_loss_coef  0.001 \

# adamw 8bit mbs=2 = 40%
# adamw 8bit mbs=4 = 89%
# adamw 4bit mbs=4 = 80% - oom

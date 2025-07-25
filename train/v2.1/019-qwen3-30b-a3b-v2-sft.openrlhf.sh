#!/usr/bin/env bash
set -eo pipefail

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"           # keep your old project
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_ASYNC_ERROR_HANDLING=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL="Qwen/Qwen3-30B-A3B"
DATA="shisa-ai/shisa-v2new-sft-shuffled"
# DATA="shisa-ai/shisa-v2-sft-shuffled"
OUT="019-qwen3-30b-a3b-v2new-sft"

# GBS = 256 - LR = 5e-7 constant
# GBS = 128 - LR = 2.5e-7
# GBS = 64 - LR = 1.25e-7

# --zero_stage      3 \
# micro_train_batch_size 16
# https://wandb.ai/augmxnt/shisa-v2.1/runs/4dhia95p
# 131.3GiB / 140.4GiB - 93.5% GPU memory
# 14/2826 [17:16<36:13:38, 46.38s/it, gpt_loss=1.06, lr=tensor(5.3821e-07), aux_loss=8.45]
# Slow - 36h / epoch ~= 300h = 900h for training run!


deepspeed --num_gpus 8 --module openrlhf.cli.train_sft \
  --pretrain        "$MODEL" \
  --dataset         "$DATA" \
  --input_key conversations \
  --apply_chat_template \
  --bf16 \
  --zero_stage      3 \
  --aux_loss_coef  0.001 \
  --optimizer  adamw_torch_4bit \
  --gradient_checkpointing \
  --flash_attn \
  --use_liger_kernel \
  --train_batch_size       128 \
  --micro_train_batch_size 16 \
  --max_len 8192 \
  --packing_samples \
  --max_epochs      3 \
  --learning_rate   1.63e-5 \
  --lr_warmup_ratio 0.05 \
  --save_path       "/data/outputs/$OUT" \
  --ckpt_path       "/data/checkpoint/$OUT" \
  --save_hf_ckpt \
  --save_steps      2826 \
  --logging_steps   1 \
  --eval_steps -1 \
  --eval_dataset "" \
  --overlap_comm \
  --use_wandb       True \
  --wandb_org        augmxnt \
  --wandb_project    shisa-v2.1 \
  --wandb_run_name   "$OUT"

# --zero_stage      2moe \ - keeps oom
#  --save_steps      2826 \  - mbs 8
#  --save_steps      11304 \ - mbs 4

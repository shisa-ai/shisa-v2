#!/usr/bin/env bash
set -eo pipefail

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"           # keep your old project
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE=0

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# MODEL="Qwen/Qwen3-30B-A3B"
MODEL="073-qwen3-30b-a3b-v2new-sft.8xMI300X.dsz3"
OUT="078-qwen3-30b-a3b-v2new-dpo405b.8xMI300X.dsz3"
DATA="dpo.shisa-v2.1.jsonl"

DPO_LR=${DPO_LR:-2.04e-7}
DPO_MBS=8

# GBS = 256 - LR = 5e-7 constant
# GBS = 128 - LR = 2.5e-7
# GBS = 64 - LR = 1.25e-7

# --zero_stage      3 \
# micro_train_batch_size 16
# https://wandb.ai/augmxnt/shisa-v2.1/runs/4dhia95p
# 131.3GiB / 140.4GiB - 93.5% GPU memory
# 14/2826 [17:16<36:13:38, 46.38s/it, gpt_loss=1.06, lr=tensor(5.3821e-07), aux_loss=8.45]
# Slow - 36h / epoch ~= 300h = 900h for training run!

echo "Starting DPO training..."
deepspeed --module openrlhf.cli.train_dpo \
  --pretrain        /data/outputs/"$MODEL" \
  --dataset         "$DATA" \
  --chosen_key      chosen \
  --rejected_key    rejected \
  --apply_chat_template \
  --bf16 \
  --zero_stage      3 \
  --zpg 2 \
  --aux_loss_coef  0.001 \
  --optimizer  adamw_torchao_4bit \
  --gradient_checkpointing \
  --attn_implementation flash_attention_2 \
  --use_liger_kernel \
  --train_batch_size       64 \
  --micro_train_batch_size 8 \
  --grad_accum_dtype bf16 \
  --max_len 4096 \
  --packing_samples \
  --max_epochs      1 \
  --learning_rate   2.04e-7 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_ratio 0.03 \
  --save_path       "/data/outputs/$OUT" \
  --logging_steps   1 \
  --eval_steps -1 \
  --eval_dataset "" \
  --overlap_comm \
  --use_wandb       True \
  --wandb_org        "$WANDB_ENTITY" \
  --wandb_project    "$WANDB_PROJECT" \
  --wandb_run_name   "$OUT"

# 41GB weights
# msb=8 OK - 166GB - 31h @ step 22+
# 5652/epoch
# msb=4 OK - 120GB - 45h @ step 24? (24h was sadly cut off at grad_accum_dtype
# 10732/epoch
# TORCH_COMPILE=0
# --zero_stage      2moe \ - keeps oom
#  --save_steps      2826 \  - mbs 8
#  --save_steps      5652 \
#  --save_steps      11304 \ - mbs 4

# - ZeRO‑2: 58.4 (weights) + 30 (grads) + 30–60 (8‑bit states) + activations → >96 GB even at micro=1.
# --zero_stage      2moe \
#
# - ZeRO‑3 with bf16 grads + partitioned 8‑bit states: ~80–84 GB at micro=1, 8k seq → feasible.
# --zero_stage      2moe \
# --max_len 8192 \
# --micro_train_batch_size 16 \
# --ring_attn_size 2
# --max_len 4096 (first), then 8192 if stable
# --grad_accum_dtype bf16

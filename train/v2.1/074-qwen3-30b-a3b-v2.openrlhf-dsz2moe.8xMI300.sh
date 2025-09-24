#!/usr/bin/env bash
set -eo pipefail

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"           # keep your old project
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE=0

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# MODEL="Qwen/Qwen3-30B-A3B"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
DATA="shisa-ai/shisa-v2new-sft-shuffled"
# DATA="shisa-ai/shisa-v2-sft-shuffled"
OUT="074-qwen3-30b-a3b-v2new-sft.8xMI300X.dsz2moe"

# GBS = 256 - LR = 5e-7 constant
# GBS = 128 - LR = 2.5e-7
# GBS = 64 - LR = 1.25e-7

# --zero_stage      3 \
# micro_train_batch_size 16
# https://wandb.ai/augmxnt/shisa-v2.1/runs/4dhia95p
# 131.3GiB / 140.4GiB - 93.5% GPU memory
# 14/2826 [17:16<36:13:38, 46.38s/it, gpt_loss=1.06, lr=tensor(5.3821e-07), aux_loss=8.45]
# Slow - 36h / epoch ~= 300h = 900h for training run!
# ZeRO-2 replicates ~57GB of weights/GPU; dropping micro batch to 2 keeps total <160GB on MI300X.

deepspeed --num_gpus 8 --module openrlhf.cli.train_sft \
  --pretrain        "$MODEL" \
  --dataset         "$DATA" \
  --input_key conversations \
  --apply_chat_template \
  --bf16 \
  --zero_stage      2moe \
  --zpg 2 \
  --aux_loss_coef  0.001 \
  --optimizer  adamw_torchao_4bit \
  --gradient_checkpointing \
  --attn_implementation flash_attention_2 \
  --use_liger_kernel \
  --train_batch_size       128 \
  --micro_train_batch_size 2 \
  --grad_accum_dtype bf16 \
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
  --wandb_org        "$WANDB_ENTITY" \
  --wandb_project    "$WANDB_PROJECT" \
  --wandb_run_name   "$OUT"

# ZeRO-2 est usage on MI300X (192GB):
#   weights ~57GB + grads/states ~76GB => ~133GB base before activations.
#   activations add ~12-13GB per micro batch at 8k seq.
#   mbs=2 -> ~158GB, safe; mbs=3 -> ~171GB; mbs=4 -> ~183GB (tight, try only after confirming headroom).

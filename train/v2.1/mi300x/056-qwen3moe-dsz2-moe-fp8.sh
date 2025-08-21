#!/usr/bin/env bash
set -eo pipefail

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"           # keep your old project
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_ASYNC_ERROR_HANDLING=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TORCH_COMPILE=0

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ORIG_MODEL=${ORIG_MODEL:-"trl-internal-testing/tiny-Qwen3MoeForCausalLM"}
SFT_CKPT=${SFT_CKPT:-"056-qwen3moe-adam_offload"}
SFT_DATA="sft.shisa-v2-new.jsonl"
SFT_LR=${SFT_LR:-1e-4}
SFT_MBS=16

export OPTIMIZE_EPILOGUE=1
export TORCH_NCCL_HIGH_PRIORITY=1
export GPU_MAX_HW_QUEUES=2

# --zero_stage      3 \
# micro_train_batch_size 16
# https://wandb.ai/augmxnt/shisa-v2.1/runs/4dhia95p
# 131.3GiB / 140.4GiB - 93.5% GPU memory
# 14/2826 [17:16<36:13:38, 46.38s/it, gpt_loss=1.06, lr=tensor(5.3821e-07), aux_loss=8.45]
# Slow - 36h / epoch ~= 300h = 900h for training run!

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
  --zero_stage      2moe \
  --zpg 2 \
  --aux_loss_coef  0.001 \
  --optimizer paged_adamw_8bit \
  --train_batch_size       128 \
  --micro_train_batch_size $SFT_MBS \
  --max_len 8192 \
  --packing_samples \
  --max_epochs      1 \
  --learning_rate   $SFT_LR \
  --lr_warmup_ratio 0.03 \
  --save_path       "$SFT_CKPT" \
  --logging_steps   1 \
  --eval_steps   -1 \
  --overlap_comm \
  --use_wandb       True \
  --wandb_org        augmxnt \
  --wandb_project    shisa-v2.1 \
  --wandb_run_name   "$SFT_CKPT-mi300x"

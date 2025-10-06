#!/usr/bin/env bash
set -eo pipefail

# OpenRLHF DPO fine-tune for Shisa v2.1 Qwen3-30B-A3B (ZeRO-3 MoE)

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE=0

# Optional: limit visible GPUs
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Base SFT checkpoint and output tag
MODEL="094-qwen3-30b-a3b-v2.1-sft/checkpoint-1188"
OUT="097-qwen3-30b-a3b-v2.1-dpo.openrlhf-dsz3.8xMI300"
DATA="dpo.shisa-v2.1.jsonl"

# Target global batch size 64 (matching Axolotl 096 LR schedule)
DPO_GBS=64
DPO_MBS=${DPO_MBS:-8}  # micro per GPU (adjust to meet memory); will be clipped to divide GBS

# Deepspeed ZeRO stage (set ZERO_STAGE=2moe for ZeRO-2 MoE config)
ZERO_STAGE=${ZERO_STAGE:-2moe}

if (( DPO_GBS % 8 != 0 )); then
  echo "[WARN] DPO_GBS (${DPO_GBS}) not divisible by data parallel degree (8); overriding to 64."
  DPO_GBS=64
fi

# Ensure micro divides global batch (fallback to 8 if not)
if (( DPO_GBS % DPO_MBS != 0 )); then
  echo "[WARN] micro batch ${DPO_MBS} does not divide global batch ${DPO_GBS}; resetting micro to 8."
  DPO_MBS=8
fi

DPO_LR=${DPO_LR:-2.04e-7}

cat <<INFO
=== DPO run summary ===
Base model:    /data/outputs/${MODEL}
Output dir:    /data/outputs/${OUT}
Dataset:       ${DATA}
ZeRO stage:    ${ZERO_STAGE}
Micro batch:   ${DPO_MBS}
Train batch:   ${DPO_GBS}
Learning rate: ${DPO_LR}
======================
INFO

deepspeed --module openrlhf.cli.train_dpo \
  --pretrain        /data/outputs/${MODEL} \
  --dataset         "${DATA}" \
  --chosen_key      chosen \
  --rejected_key    rejected \
  --apply_chat_template \
  --bf16 \
  --zero_stage      ${ZERO_STAGE} \
  --zpg             2 \
  --aux_loss_coef   0.001 \
  --optimizer       adamw_torchao_4bit \
  --gradient_checkpointing \
  --attn_implementation flash_attention_2 \
  --use_liger_kernel \
  --train_batch_size       ${DPO_GBS} \
  --micro_train_batch_size ${DPO_MBS} \
  --grad_accum_dtype bf16 \
  --max_len          4096 \
  --packing_samples \
  --max_epochs       1 \
  --learning_rate    ${DPO_LR} \
  --lr_scheduler     constant_with_warmup \
  --lr_warmup_ratio  0.03 \
  --save_path       "/data/outputs/${OUT}" \
  --logging_steps    1 \
  --eval_steps      -1 \
  --eval_dataset    "" \
  --overlap_comm \
  --use_wandb       True \
  --wandb_org        "${WANDB_ENTITY}" \
  --wandb_project    "${WANDB_PROJECT}" \
  --wandb_run_name   "${OUT}"

# Notes:
# - Adjust DPO_MBS to 8 if memory allows (GBS 64 -> LR 2.04e-7).
# - Consider --max_len 3584 if prompt budget can be reduced further.
# - Keep PYTORCH_CUDA_ALLOC_CONF exported to avoid ROCm fragmentation.

#!/usr/bin/env bash
set -euo pipefail

# export HSA_OVERRIDE_GFX_VERSION=9.4.2
export NCCL_P2P_DISABLE=1                # MI300X P2P still WIP
export TF32_OVERRIDE=0                   # keep math in BF16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:true

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT=shisa-v2.1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export HF_HUB_ENABLE_HF_TRANSFER=1

# ─── user‑tunable ───────────────────────────────────────────────
ORIG_MODEL=${ORIG_MODEL:-"Rakuten/RakutenAI-2.0-mini-instruct"}
SFT_CKPT=${SFT_CKPT:-"036-rakuten-2.0-mini-instruct-1.5b-v2new-sft-unsloth"}
DPO_CKPT=${DPO_CKPT:-"037-rakuten-2.0-mini-instruct-1.5b-v2new-dpo405b-unsloth"}
SFT_LR=${SFT_LR:-2.31e-5}
DPO_LR=${DPO_LR:-2.89e-7}
# ────────────────────────────────────────────────────────────────

# python unsloth-train-sft.py \
# 	  --base_model  "$ORIG_MODEL" \
# 	  --output_dir  "$SFT_CKPT" \
# 	  --lr          "$SFT_LR" \
#           --bs          128

python unsloth-train-dpo.py \
	  --base_model  "$SFT_CKPT" \
	  --output_dir  "$DPO_CKPT" \
	  --lr          "$DPO_LR" \
          --bs          64

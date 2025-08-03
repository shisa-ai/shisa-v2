#!/usr/bin/env bash
set -euo pipefail

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_ASYNC_ERROR_HANDLING=1

ORIG_MODEL=${ORIG_MODEL:-"qwen/Qwen3-30B-A3B"}
SFT_CKPT=${SFT_CKPT:-"054-qwen3moe-megatron"}
DATA_PREFIX=${DATA_PREFIX:-"/workspace/data/sft_dataset_text_document"}
TOKENIZER_JSON=${TOKENIZER_JSON:-"/workspace/tokenizer/qwen3-30b/tokenizer.json"}

export DATA_PATH="$DATA_PREFIX"
export LOAD_PATH="$ORIG_MODEL"
export SAVE_PATH="$SFT_CKPT"

export TEE_OUTPUT=1
export MBS=2
export GBS=128
export TP_SIZE=1
export PP_SIZE=1
export EP_SIZE=8
export ETP_SIZE=1
export AC=full
export PR=bf16
export SEQ_LENGTH=8192

cd /workspace/Megatron-LM
bash examples/llama/train_llama3.sh

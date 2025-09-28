#!/bin/bash

# Llama 3.2 1B SFT training script for the MegaBlocks ROCm environment.

set -euo pipefail

EPOCHS=${EPOCHS:-3}
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/workspace/shisa-v2.1/llama3.2-1b/checkpoints}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
DEFAULT_RUN_NAME="llama3.2_1b_${RUN_TIMESTAMP}"
RUN_NAME="${1:-${DEFAULT_RUN_NAME}}"
MICRO_BATCH_SIZE="${2:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-131072}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-8192}"

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/llama3.2-1b/data}"
DATA_DIR=${DATA_DIR%/}
DATA_PREFIX="${DATA_PREFIX:-sft.shisa-v2.1_text_document}"
EVALS_PER_EPOCH=${EVALS_PER_EPOCH:-4}
MODEL_ID="${BASE_HF_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"

mkdir -p "${CHECKPOINT_ROOT}"

DATA_IDX_FILE="${DATA_DIR}/${DATA_PREFIX}.idx"
NUM_SAMPLES=0
if [[ -f "${DATA_IDX_FILE}" ]]; then
    readarray -t IDX_META < <(DATA_IDX_FILE="${DATA_IDX_FILE}" python3 - <<'PY'
import os
import struct

path = os.environ['DATA_IDX_FILE']
with open(path, 'rb') as f:
    header = f.read(9)
    if not header.startswith(b'MMIDIDX'):
        raise SystemExit(f"Unexpected .idx header: {header!r}")
    _version = struct.unpack('<Q', f.read(8))[0]
    _dtype_code = struct.unpack('<B', f.read(1))[0]
    sequence_count = struct.unpack('<Q', f.read(8))[0]
    document_count = struct.unpack('<Q', f.read(8))[0]

actual_docs = document_count - 1 if document_count > 0 else sequence_count
if actual_docs <= 0:
    raise SystemExit("Parsed dataset size is not positive")

print(sequence_count)
print(actual_docs)
PY
)

    SEQUENCE_COUNT=${IDX_META[0]:-0}
    DOCUMENT_COUNT=${IDX_META[1]:-0}
    NUM_SAMPLES=${DOCUMENT_COUNT}
    if [[ ${NUM_SAMPLES} -le 0 ]]; then
        NUM_SAMPLES=${SEQUENCE_COUNT}
    fi
else
    echo "Warning: Data index file not found at ${DATA_IDX_FILE}. Using default dataset size of 10000 samples."
    NUM_SAMPLES=10000
fi

TOTAL_SAMPLES=$((NUM_SAMPLES * EPOCHS))
TRAINING_STEPS=$(( (TOTAL_SAMPLES + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE ))
EVAL_INTERVAL=$(( TRAINING_STEPS / (EPOCHS * EVALS_PER_EPOCH) ))
[[ ${EVAL_INTERVAL} -lt 1 ]] && EVAL_INTERVAL=1

if [[ ${EPOCHS} -gt 0 ]]; then
    SAVE_INTERVAL_STEPS=$(( TRAINING_STEPS / EPOCHS ))
    [[ ${SAVE_INTERVAL_STEPS} -lt 1 ]] && SAVE_INTERVAL_STEPS=1
else
    SAVE_INTERVAL_STEPS=${TRAINING_STEPS}
fi

SAVE_PATH="${CHECKPOINT_ROOT}/${RUN_NAME}"
if [[ -d "${SAVE_PATH}" ]]; then
    if [[ "${OVERWRITE_CHECKPOINTS:-0}" == "1" ]]; then
        echo "Warning: checkpoint directory ${SAVE_PATH} exists. Removing because OVERWRITE_CHECKPOINTS=1."
        rm -rf "${SAVE_PATH}"
    else
        echo "ERROR: Checkpoint directory ${SAVE_PATH} already exists. Set OVERWRITE_CHECKPOINTS=1 to replace it."
        exit 1
    fi
fi
mkdir -p "${SAVE_PATH}"

TOKENIZER_DIR="${DATA_DIR}"
if [[ ! -f "${TOKENIZER_DIR}/tokenizer.json" ]]; then
    echo "ERROR: Expected Hugging Face tokenizer files in ${TOKENIZER_DIR}. Run 02-generate.sh first." >&2
    exit 1
fi

# Ensure base Megatron checkpoint is available when not explicitly provided
BASE_CKPT_DIR="${BASE_CKPT_DIR:-${SCRIPT_DIR}/llama3.2-1b/base_tp8_pp1}"
base_checkpoint_ready() {
    [[ -f "${BASE_CKPT_DIR}/latest_checkpointed_iteration.txt" ]] && return 0
    compgen -G "${BASE_CKPT_DIR}/iter_*" >/dev/null 2>&1
}
if [[ -z "${INIT_CHECKPOINT}" ]]; then
    if ! base_checkpoint_ready; then
        echo "Base Megatron checkpoint not found under ${BASE_CKPT_DIR}; creating one from ${MODEL_ID}"
        HF_MODEL_DIR="${HF_MODEL_DIR:-${SCRIPT_DIR}/llama3.2-1b/hf_snapshot}"
        if [[ ! -f "${HF_MODEL_DIR}/config.json" ]]; then
            echo "Downloading Hugging Face weights to ${HF_MODEL_DIR}"
            MODEL_ID="${MODEL_ID}" HF_MODEL_DIR="${HF_MODEL_DIR}" python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
target = Path(os.environ["HF_MODEL_DIR"]).resolve()
target.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=model_id,
    local_dir=target,
    local_dir_use_symlinks=False,
)
PY
        fi

        rm -rf "${BASE_CKPT_DIR}"
        mkdir -p "${BASE_CKPT_DIR}"

        CONVERT_TRANSFORMER_IMPL="${CONVERT_TRANSFORMER_IMPL:-local}"
        CONVERT_TP_SIZE="${CONVERT_TP_SIZE:-1}"
        CONVERT_PP_SIZE="${CONVERT_PP_SIZE:-1}"
        CONVERT_CMD=(
            python3 /workspace/Megatron-LM/tools/checkpoint/convert.py
            --model-type GPT
            --loader llama_mistral
            --checkpoint-type hf
            --model-size ${CONVERT_MODEL_SIZE:-llama3-8Bf}
            --load-dir ${HF_MODEL_DIR}
            --tokenizer-model ${DATA_DIR}
            --saver mcore
            --save-dir ${BASE_CKPT_DIR}
            --target-tensor-parallel-size ${CONVERT_TP_SIZE}
            --target-pipeline-parallel-size ${CONVERT_PP_SIZE}
            --megatron-path /workspace/Megatron-LM
            --loader-transformer-impl ${CONVERT_TRANSFORMER_IMPL}
        )
        if [[ "${CONVERT_SEQUENCE_PARALLEL:-}" == "1" ]] || [[ ${CONVERT_TP_SIZE} -gt 1 ]]; then
            CONVERT_CMD+=(--sequence-parallel)
        fi

        "${CONVERT_CMD[@]}"
        if ! base_checkpoint_ready; then
            echo "ERROR: Megatron checkpoint conversion did not produce an iteration directory in ${BASE_CKPT_DIR}." >&2
            exit 1
        fi
    fi
    INIT_CHECKPOINT=${BASE_CKPT_DIR}
fi

DATA_PATH="${DATA_DIR}/${DATA_PREFIX}"

LR="${LR:-2.83e-5}"
MIN_LR="${MIN_LR:-1e-6}"

echo "=== MegaBlocks Llama 3.2 1B Training ==="
echo "Model ID: ${MODEL_ID}"
echo "Run name: ${RUN_NAME}"
echo "Epochs: ${EPOCHS}"
echo "Samples per epoch: ${NUM_SAMPLES}"
echo "Total samples: ${TOTAL_SAMPLES}"
echo "Global batch size: ${GLOBAL_BATCH_SIZE}"
echo "Micro batch size: ${MICRO_BATCH_SIZE}"
echo "Training steps: ${TRAINING_STEPS}"
echo "Eval interval: ${EVAL_INTERVAL}"
echo "Save interval (steps): ${SAVE_INTERVAL_STEPS}"
echo "Checkpoint root: ${CHECKPOINT_ROOT}"
echo "Seq length: ${SEQ_LENGTH}"
echo "Max position embeddings: ${MAX_POSITION_EMBEDDINGS}"
echo "FFN hidden size: ${FFN_HIDDEN_SIZE}"
[[ -n "${INIT_CHECKPOINT}" ]] && echo "Init checkpoint: ${INIT_CHECKPOINT}"

declare -a TORCH_ARGS
MASTER_PORT_DEFAULT=$((6000 + (RANDOM % 1000)))
TORCH_ARGS=(
    --nproc_per_node "${NPROC_PER_NODE:-8}"
    --nnodes "${NNODES:-1}"
    --node_rank "${NODE_RANK:-0}"
    --master_addr "${MASTER_ADDR:-127.0.0.1}"
    --master_port "${MASTER_PORT:-${MASTER_PORT_DEFAULT}}"
)

MODEL_ARGS="\
--num-layers 16 \
--hidden-size 2048 \
--ffn-hidden-size ${FFN_HIDDEN_SIZE} \
--num-attention-heads 32 \
--group-query-attention \
--num-query-groups ${NUM_QUERY_GROUPS:-8} \
--seq-length ${SEQ_LENGTH} \
--max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
--position-embedding-type rope \
--rotary-base 500000 \
--rotary-percent 1.0 \
--use-rope-scaling \
--untie-embeddings-and-output-weights \
--swiglu \
--disable-bias-linear \
--normalization RMSNorm"

TRAINING_ARGS="\
--micro-batch-size ${MICRO_BATCH_SIZE} \
--global-batch-size ${GLOBAL_BATCH_SIZE} \
--train-iters ${TRAINING_STEPS} \
--lr ${LR} \
--min-lr ${MIN_LR} \
--lr-decay-style cosine \
--lr-decay-iters ${TRAINING_STEPS} \
--lr-warmup-fraction 0.01 \
--clip-grad 1.0 \
--weight-decay 0.01 \
--init-method-std 0.02"

DATA_ARGS="\
--data-path ${DATA_PATH} \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model ${TOKENIZER_DIR} \
--make-vocab-size-divisible-by 128 \
--split 995,5,0"

COMPUTE_ARGS="\
--bf16 \
--no-async-tensor-model-parallel-allreduce \
--no-gradient-accumulation-fusion"

CHECKPOINT_ARGS="\
--save ${SAVE_PATH} \
--save-interval ${SAVE_INTERVAL_STEPS}"

LOG_ARGS="\
--log-interval 1 \
--eval-interval ${EVAL_INTERVAL} \
--eval-iters 20 \
--wandb-project ${WANDB_PROJECT:-shisa-v2-megablocks} \
--wandb-exp-name ${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"

EXTRA_ARGS="${EXTRA_ARGS:-}"
if [[ -n "${INIT_CHECKPOINT}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --load ${INIT_CHECKPOINT} --no-load-optim --no-load-rng"
fi

set -x
cd /workspace/Megatron-LM

torchrun "${TORCH_ARGS[@]}" \
    /workspace/shisa-v2.1/run_pretrain_with_patch.py \
    ${MODEL_ARGS} \
    ${TRAINING_ARGS} \
    ${DATA_ARGS} \
    ${COMPUTE_ARGS} \
    ${CHECKPOINT_ARGS} \
    ${LOG_ARGS} \
    ${EXTRA_ARGS}

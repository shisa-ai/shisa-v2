#!/bin/bash

# GPT-2 125M training script for ROCm 7.0 MegaBlocks environment
# This script is optimized for the ROCm 7.0 PyTorch training image

# Training configuration
EPOCHS=3
GLOBAL_BATCH_SIZE=512
EVALS_PER_EPOCH=${EVALS_PER_EPOCH:-4}
SAVE_PATH="/workspace/project/checkpoints"
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
DATA_PREFIX="${DATA_PREFIX:-sft.shisa-v2.1_text_document}"

# Calculate training parameters based on data size and epochs
# This will be calculated automatically from the data
DATA_IDX_FILE="${DATA_DIR}/${DATA_PREFIX}.idx"
if [[ -f "${DATA_IDX_FILE}" ]]; then
    # Extract number of sequences/documents from the Megatron .idx header
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
    TOTAL_SAMPLES=$((NUM_SAMPLES * EPOCHS))
    TRAINING_STEPS=$(( (TOTAL_SAMPLES + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE ))
    EVAL_INTERVAL=$(( TRAINING_STEPS / (EPOCHS * EVALS_PER_EPOCH) ))
    [[ ${EVAL_INTERVAL} -lt 1 ]] && EVAL_INTERVAL=1
    echo "Calculated training parameters:"
    echo "  - Sequence count: ${SEQUENCE_COUNT}"
    echo "  - Document count: ${DOCUMENT_COUNT}"
    echo "  - Number of samples per epoch: ${NUM_SAMPLES}"
    echo "  - Epochs: ${EPOCHS}"
    echo "  - Total samples (epochs * samples): ${TOTAL_SAMPLES}"
    echo "  - Global batch size: ${GLOBAL_BATCH_SIZE}"
    echo "  - Training steps (ceil): ${TRAINING_STEPS}"
    echo "  - Evaluation interval (steps): ${EVAL_INTERVAL}"
else
    echo "Warning: Data index file not found at ${DATA_IDX_FILE}"
    echo "Using default values"
    NUM_SAMPLES=10000
    TOTAL_SAMPLES=$((NUM_SAMPLES * EPOCHS))
    TRAINING_STEPS=$(( (TOTAL_SAMPLES + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE ))
    EVAL_INTERVAL=$(( TRAINING_STEPS / (EPOCHS * EVALS_PER_EPOCH) ))
    [[ ${EVAL_INTERVAL} -lt 1 ]] && EVAL_INTERVAL=1
fi

echo "=== MegaBlocks GPT-2 125M Training (ROCm 7.0) ==="
echo "Epochs: ${EPOCHS}"
echo "Training samples: ${TOTAL_SAMPLES}"
echo "Training steps: ${TRAINING_STEPS}"
echo "Eval interval (steps): ${EVAL_INTERVAL}"
echo "Save path: ${SAVE_PATH}"
echo "Data directory: ${DATA_DIR}"
echo "Data prefix: ${DATA_PREFIX}"
if [[ -n "${NUM_SAMPLES:-}" ]]; then
    echo "Dataset info: ${NUM_SAMPLES} samples, ${EPOCHS} epochs"
fi
echo ""

# Create checkpoints directory in project space and clean if exists
if [[ -d "${SAVE_PATH}" ]]; then
    echo "Warning: Checkpoint directory ${SAVE_PATH} already exists. Removing..."
    rm -rf "${SAVE_PATH}"
fi
mkdir -p "${SAVE_PATH}"

##
### Pre-training for GPT2 125M parameter.
##

# Distributed hyperparameters - optimized for MI300X
DISTRIBUTED_ARGUMENTS="\
--nproc_per_node 8 \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6000"

# Model hyperparameters
MODEL_ARGUMENTS="\
--num-layers 12 \
--hidden-size 768 \
--num-attention-heads 12 \
--seq-length 1024 \
--max-position-embeddings 1024"

# Training hyperparameters - optimized for ROCm 7.0
TRAINING_ARGUMENTS="\
--micro-batch-size 64 \
--global-batch-size ${GLOBAL_BATCH_SIZE} \
--train-samples ${TOTAL_SAMPLES} \
--lr-decay-samples ${TOTAL_SAMPLES} \
--lr 0.0006 \
--min-lr 0.00006 \
--lr-decay-style cosine \
--lr-warmup-fraction 0.01 \
--clip-grad 1.0 \
--init-method-std 0.01"

# Data paths
VOCAB_FILE="${DATA_DIR}/gpt2-vocab.json"
MERGE_FILE="${DATA_DIR}/gpt2-merges.txt"
DATA_PATH="${DATA_DIR}/${DATA_PREFIX}"

DATA_ARGUMENTS="\
--data-path ${DATA_PATH} \
--vocab-file ${VOCAB_FILE} \
--merge-file ${MERGE_FILE} \
--make-vocab-size-divisible-by 1024 \
--split 995,5,0"

# Compute arguments - optimized for ROCm 7.0
COMPUTE_ARGUMENTS="\
--bf16 \
--no-async-tensor-model-parallel-allreduce \
--no-gradient-accumulation-fusion"

# Checkpoint arguments
# Calculate save interval (save at end of each epoch based on samples)
SAVE_INTERVAL_SAMPLES=$((TOTAL_SAMPLES / EPOCHS))
[[ ${SAVE_INTERVAL_SAMPLES} -lt 1000 ]] && SAVE_INTERVAL_SAMPLES=1000  # Minimum save interval

CHECKPOINT_ARGUMENTS="\
--save-interval ${SAVE_INTERVAL_SAMPLES} \
--save ${SAVE_PATH}"

# Logging / evaluation arguments
LOGGING_ARGUMENTS="\
--log-interval 1 \
--eval-interval ${EVAL_INTERVAL} \
--eval-iters 20"

# Wandb logging arguments
WANDB_ARGUMENTS="\
--wandb-project ${WANDB_PROJECT:-shisa-v2-megablocks} \
--wandb-exp-name dense_gpt2_125m_$(date +%Y%m%d_%H%M%S) \
--log-params-norm \
--log-num-zeros-in-grad"

echo "Starting training..."
echo "Logs will be saved to: ${SAVE_PATH}/train.log"
echo "Save interval: ${SAVE_INTERVAL_SAMPLES} samples"
echo "Training start time: $(date)"
TRAINING_START_TIME=$(date +%s)

cd /workspace/Megatron-LM

torchrun ${DISTRIBUTED_ARGUMENTS} \
       /workspace/project/run_pretrain_with_patch.py \
       ${MODEL_ARGUMENTS} \
       ${TRAINING_ARGUMENTS} \
       ${DATA_ARGUMENTS} \
       ${COMPUTE_ARGUMENTS} \
       ${CHECKPOINT_ARGUMENTS} \
       ${LOGGING_ARGUMENTS} \
       ${WANDB_ARGUMENTS} |& tee ${SAVE_PATH}/train.log

# Calculate and display training statistics
TRAINING_END_TIME=$(date +%s)
TRAINING_DURATION=$((TRAINING_END_TIME - TRAINING_START_TIME))
TRAINING_HOURS=$((TRAINING_DURATION / 3600))
TRAINING_MINUTES=$(((TRAINING_DURATION % 3600) / 60))
TRAINING_SECONDS=$((TRAINING_DURATION % 60))

echo ""
echo "=== Training completed! ==="
echo "Training end time: $(date)"
echo "Total training duration: ${TRAINING_HOURS}h ${TRAINING_MINUTES}m ${TRAINING_SECONDS}s"
echo "Training steps completed: ${TRAINING_STEPS}"
echo "Training samples processed: ${TOTAL_SAMPLES}"
echo "Epochs completed: ${EPOCHS}"
if [[ -n "${NUM_SAMPLES:-}" ]]; then
    echo "Dataset samples: ${NUM_SAMPLES}"
    echo "Samples per second: $(( TOTAL_SAMPLES / TRAINING_DURATION ))"
fi
echo "Checkpoints saved to: ${SAVE_PATH}"
echo "Training log: ${SAVE_PATH}/train.log"

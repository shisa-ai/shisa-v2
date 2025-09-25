#!/bin/bash

# GPT-2 125M MoE training script for ROCm 7.0 MegaBlocks environment
# This script uses Mixture of Experts and is optimized for the ROCm 7.0 PyTorch training image

EXP_DIR="${1:-moe_experiment}"
TRAINING_STEPS="${2:-2000}"
NUM_EXPERTS="${3:-64}"
CAPACITY_FACTOR="${4:-1}"
TOP_K="${5:-1}"
LOSS_WEIGHT="${6:-0.1}"
BATCH_SIZE="${7:-32}"

SAVE_PATH="/workspace/project/checkpoints/${EXP_DIR}"
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
DATA_PREFIX="${DATA_PREFIX:-sft.shisa-v2.1_text_document}"

echo "=== MegaBlocks GPT-2 125M MoE Training (ROCm 7.0) ==="
echo "Experiment directory: ${EXP_DIR}"
echo "Training steps: ${TRAINING_STEPS}"
echo "Number of experts: ${NUM_EXPERTS}"
echo "Top-K: ${TOP_K}"
echo "Save path: ${SAVE_PATH}"
echo "Data directory: ${DATA_DIR}"
echo "Data prefix: ${DATA_PREFIX}"
echo ""

# Create experiment directory in project space
mkdir -p "${SAVE_PATH}"

##
### Pre-training for MoE GPT-2 125M parameter.
##

# MoE hyperparameters
# Note: These argument names have been updated for current Megatron-LM compatibility:
# - Changed from --moe-num-experts to --num-experts
# - Changed from --moe-capacity-factor to --moe-expert-capacity-factor
# - Changed from --moe-loss-weight to --moe-aux-loss-coeff
# - Changed from --moe-top-k to --moe-router-topk
# - Removed --moe-expert-model-parallelism (not supported)
# - Added --moe-token-dispatcher-type=alltoall (required for capacity factor)
# - Added --moe-router-pre-softmax (required when top-k=1 for numerical stability)
# - Added --disable-bias-linear (MoE alltoall dispatcher doesn't support bias)
MOE_ARGUMENTS="\
--num-experts=${NUM_EXPERTS} \
--moe-expert-capacity-factor=${CAPACITY_FACTOR} \
--moe-aux-loss-coeff=${LOSS_WEIGHT} \
--moe-router-topk=${TOP_K} \
--moe-token-dispatcher-type=alltoall \
--moe-router-pre-softmax \
--disable-bias-linear"

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

# Training hyperparameters - optimized for MoE
TRAINING_ARGUMENTS="\
--micro-batch-size ${BATCH_SIZE} \
--global-batch-size 512 \
--train-iters ${TRAINING_STEPS} \
--lr-decay-iters ${TRAINING_STEPS} \
--lr 0.00015 \
--min-lr 0.00001 \
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
--split 969,30,1"

# Compute arguments - optimized for ROCm 7.0 MoE
# Note: Using bf16 instead of fp16 for better stability on MI300X
# Disabling async and fusion flags for ROCm compatibility with MoE layers
COMPUTE_ARGUMENTS="\
--bf16 \
--no-async-tensor-model-parallel-allreduce \
--no-gradient-accumulation-fusion"

# Checkpoint arguments
CHECKPOINT_ARGUMENTS="\
--save-interval 2000 \
--save ${SAVE_PATH}"

# Evaluation arguments
EVALUATION_ARGUMENTS="\
--eval-iters 100 \
--log-interval 100 \
--eval-interval 1000"

# Wandb logging arguments (uncomment to enable)
# WANDB_ARGUMENTS="\
# --wandb-project megablocks_moe \
# --wandb-exp-name ${EXP_DIR} \
# --log-params-norm \
# --log-num-zeros-in-grad \
# --log-validation-ppl-to-tensorboard"

echo "Starting MoE training..."
echo "MoE Configuration:"
echo "  - Experts: ${NUM_EXPERTS}"
echo "  - Top-K: ${TOP_K}"
echo "  - Capacity Factor: ${CAPACITY_FACTOR}"
echo "  - Loss Weight: ${LOSS_WEIGHT}"
echo ""
echo "Logs will be saved to: ${SAVE_PATH}/train.log"

cd /workspace/Megatron-LM

torchrun ${DISTRIBUTED_ARGUMENTS} \
       pretrain_gpt.py \
       ${MOE_ARGUMENTS} \
       ${MODEL_ARGUMENTS} \
       ${TRAINING_ARGUMENTS} \
       ${DATA_ARGUMENTS} \
       ${COMPUTE_ARGUMENTS} \
       ${CHECKPOINT_ARGUMENTS} \
       ${EVALUATION_ARGUMENTS} \
       ${WANDB_ARGUMENTS:-} |& tee ${SAVE_PATH}/train.log

echo ""
echo "=== MoE Training completed! ==="
echo "Checkpoints saved to: ${SAVE_PATH}"
echo "Training log: ${SAVE_PATH}/train.log"
echo ""
echo "Usage for different configurations:"
echo "  ./$(basename $0) experiment_name [steps] [experts] [capacity] [top_k] [loss_weight] [batch_size]"
echo "  Example: ./$(basename $0) my_moe_run 5000 128 2 2 0.05 16"

#!/bin/bash

# GPT-2 125M training script for ROCm 7.0 MegaBlocks environment
# This script is optimized for the ROCm 7.0 PyTorch training image

TRAINING_STEPS=2000
SAVE_PATH="/workspace/project/checkpoints"

echo "=== MegaBlocks GPT-2 125M Training (ROCm 7.0) ==="
echo "Training steps: ${TRAINING_STEPS}"
echo "Save path: ${SAVE_PATH}"
echo ""

# Create checkpoints directory in project space
mkdir -p ${SAVE_PATH}

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
--global-batch-size 512 \
--train-iters ${TRAINING_STEPS} \
--lr-decay-iters ${TRAINING_STEPS} \
--lr 0.0006 \
--min-lr 0.00006 \
--lr-decay-style cosine \
--lr-warmup-fraction 0.01 \
--clip-grad 1.0 \
--init-method-std 0.01"

# Data paths
VOCAB_FILE="/workspace/project/gpt2-vocab.json"
MERGE_FILE="/workspace/project/gpt2-merges.txt"
DATA_PATH="/workspace/project/my-gpt2_text_document"

DATA_ARGUMENTS="\
--data-path ${DATA_PATH} \
--vocab-file ${VOCAB_FILE} \
--merge-file ${MERGE_FILE} \
--make-vocab-size-divisible-by 1024 \
--split 969,30,1"

# Compute arguments - optimized for ROCm 7.0
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

echo "Starting training..."
echo "Logs will be saved to: ${SAVE_PATH}/train.log"

cd /workspace/Megatron-LM

torchrun ${DISTRIBUTED_ARGUMENTS} \
       pretrain_gpt.py \
       ${MODEL_ARGUMENTS} \
       ${TRAINING_ARGUMENTS} \
       ${DATA_ARGUMENTS} \
       ${COMPUTE_ARGUMENTS} \
       ${CHECKPOINT_ARGUMENTS} \
       ${EVALUATION_ARGUMENTS} |& tee ${SAVE_PATH}/train.log

echo ""
echo "=== Training completed! ==="
echo "Checkpoints saved to: ${SAVE_PATH}"
echo "Training log: ${SAVE_PATH}/train.log"
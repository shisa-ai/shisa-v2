#!/bin/bash

#SBATCH --job-name=sharegpt-base-filtered
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=/fsx/user02/logs/%x-%j.out
#SBATCH --error=/fsx/user02/logs/%x-%j.err

AXOLOTL_CFG=sharegpt-base-filtered.yaml

echo "START TIME: $(date)"

set -eo pipefail
set -x

LOG_PATH="/fsx/user02/logs/main_log.txt"

source /fsx/user02/.bashrc
mamba activate axolotl
cd /fsx/user02/axolotl

export HF_HUB_ENABLE_HF_TRANSFER=1
export ACCELERATE_LOG_LEVEL=info
export TRANSFORMERS_VERBOSITY=info
export NCCL_ASYNC_ERROR_HANDLING=1

export WANDB_ENTITY='augmxnt'
export WANDB_PROJECT='shisa-v2'

GPUS_PER_NODE=8
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
NNODES=$SLURM_JOB_NUM_NODES  # Correct variable name
NODE_RANK=$SLURM_PROCID
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CMD="-m axolotl.cli.train $AXOLOTL_CFG"

LAUNCHER="accelerate launch \
    --multi_gpu \
    --dynamo_backend no \
    --mixed_precision bf16 \
    --num_machines $NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role $SLURMD_NODENAME: \
    --rdzv_conf rdzv_backend=c10d \
    --max_restarts 0 \
    --tee 3 \
"

SRUN_ARGS="--wait=60 --kill-on-bad-exit=1"

# We do this once...
# CUDA_VISIBLE_DEVICES='' python -m axolotl.cli.preprocess llama3-70b-fft.yaml; 
clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER $CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"

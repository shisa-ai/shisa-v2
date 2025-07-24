#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME=megatron-lm
IMAGE="rocm/megatron-lm:latest"

TRAIN_DIR="$HOME/shisa-v2/train/v2.1/mi300x"
HF_CACHE_DIR="$HOME/.cache/huggingface"

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Attaching to running container ${CONTAINER_NAME}..."
    exec docker exec -it ${CONTAINER_NAME} bash
else
    echo "Starting new container ${CONTAINER_NAME} from ${IMAGE}..."
    exec docker run --rm -it \
        --name ${CONTAINER_NAME} \
        --ipc=host --network=host --privileged \
        --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
        --cap-add CAP_SYS_ADMIN --cap-add SYS_PTRACE \
        --group-add render \
        --security-opt seccomp=unconfined \
        --mount type=bind,src="${TRAIN_DIR}",target=/workspace/train \
        --mount type=bind,src="${HF_CACHE_DIR}",target=/root/.cache/huggingface \
        ${IMAGE} bash
fi


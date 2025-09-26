#!/bin/bash

# Run ROCm 7.0 PyTorch training container
# Mounts current directory and drops into shell

echo "=== Starting ROCm 7.0 Container ==="
echo "Image: rocm/7.0:rocm7.0_pytorch_training_instinct_20250915"
echo ""

CURRENT_DIR=$(pwd)
echo "Mounting current directory: ${CURRENT_DIR} -> /workspace/project"
echo "Starting interactive container..."

docker run -it --rm \
  --privileged \
  -v "${CURRENT_DIR}":/workspace/project \
  -v /root/.cache:/root/.cache \
  -v /root/.netrc:/root/.netrc:ro \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --name=rocm7_container_$(date +%s) \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size 16G \
  -e WANDB_PROJECT=shisa-v2-megablocks \
  -e WANDB_LOG_MODEL=false \
  -e WANDB_WATCH=false \
  rocm/7.0:rocm7.0_pytorch_training_instinct_20250915 \
  /bin/bash -c "
    echo 'Installing MegaBlocks...'
    cd /workspace/project/megablocks && python3 setup.py develop --no-deps
    echo 'MegaBlocks installation complete!'
    echo 'Starting interactive shell...'
    cd /workspace/project
    exec /bin/bash
  "

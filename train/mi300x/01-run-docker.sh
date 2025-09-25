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
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --name=rocm7_container_$(date +%s) \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size 16G \
  rocm/7.0:rocm7.0_pytorch_training_instinct_20250915 \
  /bin/bash

docker pull rocm/vllm:latest
docker rm vllm
docker run -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --shm-size 16G \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    --cap-add=SYS_PTRACE \
    -v $(pwd):/workspace \
    -v /root/.cache:/root/.cache \
    --name vllm \
    rocm/vllm:latest \
    /bin/bash

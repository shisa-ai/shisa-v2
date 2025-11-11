#!/bin/bash

# Default model if not provided
MODEL=${1:-shisa-ai/shisa-v2-llama3.1-405b}

# docker pull rocm/vllm-dev:main

docker run -it --rm --network=host --privileged --device=/dev/kfd --device=/dev/dri \
    --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v $HOME/dockerx:/dockerx \
    -v /data:/data \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    rocm/vllm-dev:main \
    python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size 8 \
    --host 0.0.0.0 \
    --port 8000


    # --env "HF_TOKEN=$HF_TOKEN" \

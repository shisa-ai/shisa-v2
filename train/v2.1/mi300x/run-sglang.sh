#!/bin/bash

# docker pull lmsysorg/sglang:v0.5.3.post3-rocm700-mi30x

docker run -it --rm --network=host --privileged --device=/dev/kfd --device=/dev/dri \
    --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v $HOME/dockerx:/dockerx \
    -v /data:/data \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    lmsysorg/sglang:v0.5.3.post3-rocm700-mi30x \
    python3 -m sglang.launch_server \
    --model-path shisa-ai/shisa-v2-llama3.1-405b \
    --tp 8 \
    --host 0.0.0.0 \
    --port 8000

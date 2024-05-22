# docker run --gpus '"all"' --rm -it winglian/axolotl:main-latest
docker run --privileged --gpus '"all"' --shm-size 10g --rm -it --name axolotl --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --mount type=bind,src="${PWD}",target
=/workspace/axolotl -v ${HOME}/.cache/huggingface:/root/.cache/huggingface winglian/axolotl:main-latest

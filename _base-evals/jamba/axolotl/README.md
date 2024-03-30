# shisa-jamba-v1 

Training Jamba on Axolotl (2024-03-30)

We start with [Peter Devine's](https://huggingface.co/ptrdvn) WandB writeup which gives us a blueprint:
* https://wandb.ai/peterd/axolotl/reports/Jamba-chat-finetuning-Airoboros-and-ShareGPT---Vmlldzo3MzUwNTc1?accessToken=syabbmhblmwslnizuwws62tig3f0op75zh2jm2owbranaziq7thbgizl78nnowgc

Instead of vast.ai, we use RunPod Community Cloud since for 4x80 cards, vast.ai only has H100s available, and it's a bit spendy.

We are able to get a RunPod of 4xA100-80 (300W PCIe) for $6.36/h.
* About 30 minutes to get up and running
* About 31h for 3 epochs of training
* At 32h, $203.52

Run @ https://wandb.ai/augmxnt/shisa-v2/runs/o830e1kw


## Axolotl
With RunPod, we can use the `winglian/axolotl-cloud:main-latest` Docker image directly:
* https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#cloud-gpu
* Setup link: https://runpod.io/gsc?template=v2ickqhz9s&ref=6i7fkpdz

## System
```
apt update
apt install byobu -y
apt install neovim -y
apt install btop htop nvtop -y
```

## Byobu
By default the docker image autoloads tmux on login. Let's switch up the `.bashrc`
```
[[ -z "$BYOBU_RUN_DIR" ]] && { byobu attach-session -t ssh_byobu || byobu new-session -s ssh_byobu; exit; }
```

## HF
```
pip install huggingface_hub
git config --global credential.helper store
huggingface-cli login

# huggingface-cli download ai21labs/Jamba-v0.1
```

## WandB
```
Setup install wandb
wandb loginp
```

## Train
```
cd /workspace
git clone git@github.com:shisa-ai/shisa-v2.git
cd /workspace/shisa-v2/_base-evals/jamba/axolotl
./train.sh
```

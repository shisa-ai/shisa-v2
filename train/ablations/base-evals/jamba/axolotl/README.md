# shisa-jamba-v1 
Training Jamba on Axolotl (2024-03-30)
- 3-epoch checkpoint: https://huggingface.co/shisa-ai/shisa-jamba-v1-checkpoint-4228
- WandB training logs: https://wandb.ai/augmxnt/shisa-v2/runs/o830e1kw

We start with [Peter Devine's](https://huggingface.co/ptrdvn) WandB writeup which gives us a blueprint:
- https://wandb.ai/peterd/axolotl/reports/Jamba-chat-finetuning-Airoboros-and-ShareGPT---Vmlldzo3MzUwNTc1?accessToken=syabbmhblmwslnizuwws62tig3f0op75zh2jm2owbranaziq7thbgizl78nnowgc

Instead of vast.ai, we use RunPod Community Cloud since for 4x80 cards, vast.ai only had H100s available when I was spinning up, and that was a bit spendy.

We were able to get a RunPod Community Cloud of 4xA100-80 (300W PCIe) for $6.36/hr in NL on fast internet.
- About 30 minutes to get up and running
- About 31h for 3 epochs of training
- At 32h, $204

A RunPod Secure Cloud 4xA100-80 (500W SXM) for $9.16/hr (US, very fast internet). This is about 44% more expensive, so the question will be if we can train that much faster.
- Using this checkout, we were able to get setup in 5-10 minutes and training in <10min
- About 26.5h for 3 epochs of training (only 17% faster, would need 21.5h for cost breakeven)
- At 27h, $247

Run @ https://wandb.ai/augmxnt/shisa-v2/runs/o830e1kw

## Axolotl
With RunPod, we can use the `winglian/axolotl-cloud:main-latest` Docker image directly:
- https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#cloud-gpu
- Direct setup link (presumably axolotl referral link, they deserve it!): https://runpod.io/gsc?template=v2ickqhz9s&ref=6i7fkpdz

## System
```
apt update
apt install byobu -y
apt install neovim -y
apt install btop dstat htop inxi nvtop -y
```

## Byobu
By default the docker image autoloads tmux on login. Let's switch up the `.bashrc`
```
sed -i 's/\[\[ -z "\$TMUX"  \]\] && { tmux attach-session -t ssh_tmux || tmux new-session -s ssh_tmux; exit; }/\[\[ -z "\$BYOBU_RUN_DIR" \]\] \&\& { byobu attach-session -t ssh_byobu || byobu new-session -s ssh_byobu; exit; }/' ~/.bashrc
```
- Relogin to switch to byobu

# Env Setup
```
echo 'HF_HOME=/workspace/hf' >> ~/.bashrc
echo 'HF_HUB_ENABLE_HF_TRANSFER=1' >> ~/.bashrc

git config --global user.email 'lhl@randomfoo.net'
git config --global user.name 'lhl'

pip install huggingface_hub
git config --global credential.helper store
huggingface-cli login

# if we want a head start...
# huggingface-cli download ai21labs/Jamba-v0.1

Setup install wandb
wandb login
```
- HF Tokens: https://huggingface.co/settings/tokens
- WandB Tokens: https://wandb.ai/settings (Danger Zone section)

## Train
```
cd /workspace
git clone git@github.com:shisa-ai/shisa-v2.git
cd /workspace/shisa-v2/_base-evals/jamba/axolotl
./train.sh
```

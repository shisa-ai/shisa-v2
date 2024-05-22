# Test results:

Initial tuning tests: https://api.wandb.ai/links/augmxnt/h4mc4dd5

# vast.ai setup

## System
```
touch ~/.no_auto_tmux
# Relogin
apt update
apt install byobu -y
apt install neovim -y
apt install btop htop nvtop -y
```
##  HF
```
pip install huggingface_hub
git config --global credential.helper store
huggingface-cli login

huggingface-cli download ai21labs/Jamba-v0.1
```

## WandB
```
pip install wandb
wandb login
```

## ML Libs
```
pip install -U torch torchaudio torchvideo
pip install transformers
pip install accelerate
pip install bitsandbytes
```

## FA2
We need to install `cuda-toolkit` to build FA2
```
apt-get install linux-headers-$(uname -r)
apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install cuda-toolkit
pip install flash_attn --no-build-isolation
```
* https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu

## Jamba
```
pip install datasets peft trl
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps packaging ninja einops xformers
pip install mamba-ssm 'causal-conv1d==1.2.0.post2'
```
* https://colab.research.google.com/drive/1qqSDIq61kw1IcWy1Jhbg6sZ9EVGnWIUv#scrollTo=GFFoj-wqjyV6

### Mamba Issues
If you run into problems, you could try manually compiling
```
pip uninstall mamba-ssm causal-conv1d
git clone https://github.com/Dao-AILab/causal-conv1d
cd causal-conv1d
python setup.py install
cd ..
git clone https://github.com/state-spaces/mamba
cd mamba
pip install .
```

## JA MT-Bench
This is a PITA since it may screw with your libs...
```
git clone https://github.com/Stability-AI/FastChat.git
cd FastChat
pip install -e ".[model_worker,llm_judge]"

# My Fork
git clone https://github.com/AUGMXNT/llm-judge
cd llm-judge
pip install -r pip install -r requirements/base.txt
pip install -r pip install -r requirements/analyze.txt

# Listen fschat, we want to be up to date
pip install -U transformers
pip install -U pydantic
pip install -U torch
```

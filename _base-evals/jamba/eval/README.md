Oops, we forgot to check in the README.md...

Let's see, spun up a 2xA100 (minimum needed for FP16/BF16 inference) on RunPod Secure Cloud.

Axolotl image had some lib incompatibilities when trying to do testing so just used RunPod PyTorch 2.1 image instead.

# System
```
apt update
apt install byobu -y
apt install neovim -y
apt install btop dstat htop inxi nvtop -y

byobu
```

# Env
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
```
- HF Tokens: https://huggingface.co/settings/tokens

# Libs
```
pip install -U torch torchaudio torchvideo
pip install transformers
pip install accelerate
pip install bitsandbytes

pip install datasets peft trl

# Had to compile...
git clone https://github.com/Dao-AILab/causal-conv1d
cd causal-conv1d
python setup.py install
cd ..
git clone https://github.com/state-spaces/mamba
cd mamba
pip install .
```

# JA MT-Bench
```
git clone https://github.com/Stability-AI/FastChat.git
cd FastChat
pip install -e ".[model_worker,llm_judge]"

# Listen fschat, we want to be up to date
pip install -U transformers
pip install -U pydantic
pip install -U torch

ln -s FastChat/fastchat/llm_judge/data
```

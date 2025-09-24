## bitsandbytes
```
# Install bitsandbytes from source
# Clone bitsandbytes repo, ROCm backend is currently enabled on multi-backend-refactor branch
git clone -b multi-backend-refactor https://github.com/bitsandbytes-foundation/bitsandbytes.git && cd bitsandbytes/

# Compile & install
apt-get install -y build-essential cmake  # install build tools dependencies, unless present
cmake -DCOMPUTE_BACKEND=hip -S .  # Use -DBNB_ROCM_ARCH="gfx90a;gfx942" to target specific gpu arch
make
pip install -e .   # `-e` for "editable" install, when developing BNB (otherwise leave that out)
```
- https://github.com/bitsandbytes-foundation/bitsandbytes/blob/b2a8a15610d696b3ed42df7af3b7109a99319ce7/docs/source/installation.mdx#multi-backend-pip

## axolotl
```
pip install --no-build-isolation axolotl
pip install -U transformers
```


## RUNS

### 30B-A3B SFT

### 073-qwen3-30b-a3b-v2new-sft.8xMI300X.dsz3
- https://wandb.ai/augmxnt/shisa-v2.1/runs/y9u9sleq
  - 36.5h - 1.2 epoch/crash
- https://wandb.ai/augmxnt/shisa-v2.1/runs/55z1aqc3
  - 68.25h - 2 epoch

Total = 105h = 840h




```
# pyaotriton
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cp aotriton/build/install_dir/lib/pyaotriton.cpython-312-x86_64-linux-gnu.so $SITE_PACKAGES/


# PyTorch?
cp aotriton/build/install_dir/lib/pyaotriton.cpython-312-x86_64-linux-gnu.so $SITE_PACKAGES/torch/lib/
cp aotriton/build/install_dir/lib/libaotriton_v2.so.0.12.0 $SITE_PACKAGES/torch/lib/
mkdir -p $SITE_PACKAGES/torch/lib/aotriton.images
cp -r aotriton/build/install_dir/lib/aotriton.images/* $SITE_PACKAGES/torch/lib/aotriton.images/

cd $SITE_PACKAGES/torch/lib && ln -sf libaotriton_v2.so.0.12.0 libaotriton_v2.so
```

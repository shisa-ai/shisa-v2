CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess fft-8b.yaml
accelerate launch -m axolotl.cli.train fft-8b.yaml --deepspeed axolotl/deepspeed_configs/zero2.json

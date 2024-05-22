accelerate launch \
  --config_file /fsx/user02/axolotl/config.yaml \
  --main_process_ip ip-10-1-0-244 \
  --main_process_port 6000 \
  --num_machines 2 \
  --machine_rank 0 \
#  python -m axolotl.cli.train llama3-70b-fft.yaml --deepspeed axolotl/deepspeed_configs/zero3_bf16.json
#  python -m axolotl.cli.train llama3-70b-fft.yaml --deepspeed axolotl/deepspeed_configs/zero3_bf16.json


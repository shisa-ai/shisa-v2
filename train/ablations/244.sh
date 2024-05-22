if [ -z "$1" ]; then
  echo "Axolotl yaml file required as an argument."
  exit 1
fi
accelerate launch \
  --num_processes 16 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_ip ip-10-1-0-244 \
  --main_process_port 6000 \
  --num_machines 2 \
  --machine_rank 0 \
  -m axolotl.cli.train $1

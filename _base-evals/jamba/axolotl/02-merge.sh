# merge the 2416 checkpoint 
python -m axolotl.cli.merge_lora train_jamba_shisa.yaml --lora_model_dir="./checkpoint-2416"
mv shisa-jamba-v1/merged ./shisa-jamba-v1-checkpoint-2416

# merge the 4228 checkpoint
python -m axolotl.cli.merge_lora train_jamba_shisa.yaml
mv shisa-jamba-v1/merged ./shisa-jamba-v1-checkpoint-4228

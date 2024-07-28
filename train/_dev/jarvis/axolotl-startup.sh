# Log into Huggingface
huggingface-cli login --token $HUGGINGFACE_TOKEN

# Download repo w/ Axolotl Configs
git clone https://github.com/shisa-ai/shisa-v2/
SHISA_DIR="$(pwd)/shisa-v2"
AXOLOTL_CFG="$SHISA_DIR/train/ablations/[BLAH].yaml"

# Run Axolotl (it should auto-download the datasets and models)
cd /home/axolotl
accelerate launch -m axolotl.cli.train examples/openllama-3b/lora.yml

# Now we want to cleanup and upload to HF
rm -rf out/checkpoints*
# Just me getting stray empty model?
rm out/model.safetensors
# Actual upload
$SHISA_DIR/train/upload-to-hf.py out/

# Can we shutdown or do we need the API? https://jarvislabs.ai/docs/api 

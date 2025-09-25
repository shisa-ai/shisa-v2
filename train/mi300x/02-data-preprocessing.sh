#!/bin/bash

# Data preprocessing script for ROCm 7.0 MegaBlocks environment
# Run this inside the ROCm 7.0 container after MegaBlocks installation

BASE_DATA_PATH=/workspace/project
cd ${BASE_DATA_PATH}

echo "=== Data Preprocessing for MegaBlocks ROCm 7.0 ==="
echo "Working directory: $(pwd)"
echo ""

echo "Obtaining dataset, vocabulary, and merge table from HuggingFace:"

# Download the oscar dataset
wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
xz -d oscar-1GB.jsonl.xz

# Download GPT-2 tokenizer files
wget --output-document=gpt2-vocab.json https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json
wget --output-document=gpt2-merges.txt https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt

echo ""
echo "Preprocessing training data:"

# Preprocess the data
time python /workspace/Megatron-LM/tools/preprocess_data.py \
    --input ${BASE_DATA_PATH}/oscar-1GB.jsonl \
    --output-prefix ${BASE_DATA_PATH}/my-gpt2 \
    --vocab-file ${BASE_DATA_PATH}/gpt2-vocab.json \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file ${BASE_DATA_PATH}/gpt2-merges.txt \
    --append-eod \
    --workers 40

echo ""
echo "=== Data preprocessing completed! ==="
echo "Generated files:"
echo "  - ${BASE_DATA_PATH}/my-gpt2_text_document.bin"
echo "  - ${BASE_DATA_PATH}/my-gpt2_text_document.idx"
echo ""
echo "These files are ready for MegaBlocks training."

cd /workspace

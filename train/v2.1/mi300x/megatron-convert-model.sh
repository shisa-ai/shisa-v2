#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <hf_model_dir> <output_dir>" >&2
    exit 1
fi

HF_MODEL=$1
OUT_DIR=$2
MEGATRON_DIR=${MEGATRON_DIR:-/workspace/Megatron-LM}

python $MEGATRON_DIR/tools/checkpoint/convert_hf_checkpoint.py \
       --input-dir "$HF_MODEL" \
       --output-dir "$OUT_DIR"

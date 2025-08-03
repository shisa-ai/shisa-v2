#!/usr/bin/env bash
set -euo pipefail

# Install MegaBlocks ROCm fork inside the Megatron-LM container.
# Run this script inside the rocm/megatron-lm Docker image.

MEGABLOCKS_DIR=${MEGABLOCKS_DIR:-$HOME/megablocks}

if [ ! -d "$MEGABLOCKS_DIR" ]; then
    git clone https://github.com/ROCm/megablocks.git "$MEGABLOCKS_DIR"
fi

pip install -U pip
pip install "$MEGABLOCKS_DIR"

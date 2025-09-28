#!/bin/bash

# Launch Qwen3-0.6B fine-tuning using the shared MegaBlocks launcher.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"

MODEL_DATA_DIR="${SCRIPT_DIR}/data"
MODEL_CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints"
mkdir -p "${MODEL_DATA_DIR}" "${MODEL_CHECKPOINT_DIR}"

DATA_DIR="${MODEL_DATA_DIR}" \
CHECKPOINT_ROOT="${MODEL_CHECKPOINT_DIR}" \
BASE_HF_MODEL="${BASE_HF_MODEL:-Qwen/Qwen3-0.6B}" \
bash "${REPO_ROOT}/03-megablocks-qwen3-0.6b.sh" "$@"

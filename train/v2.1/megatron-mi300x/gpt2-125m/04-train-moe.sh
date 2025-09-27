#!/bin/bash

# Launch MoE GPT-2 125M training with model-scoped data/checkpoint directories.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"

MODEL_DATA_DIR="${SCRIPT_DIR}/data"
MODEL_CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints"

mkdir -p "${MODEL_DATA_DIR}" "${MODEL_CHECKPOINT_DIR}"

DATA_DIR="${MODEL_DATA_DIR}" \
CHECKPOINT_ROOT="${MODEL_CHECKPOINT_DIR}" \
bash "${REPO_ROOT}/04-megablocks-moe-gpt2-125m.sh" "$@"

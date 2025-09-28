#!/bin/bash

# Thin wrapper to build the GPT-2 125M dataset inside this model folder.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"

MODEL_DATA_DIR="${SCRIPT_DIR}/data"
mkdir -p "${MODEL_DATA_DIR}"

cd "${SCRIPT_DIR}"
python3 "${REPO_ROOT}/02-generate.sft.shisa-v2.1-megablocks.py" "$@"

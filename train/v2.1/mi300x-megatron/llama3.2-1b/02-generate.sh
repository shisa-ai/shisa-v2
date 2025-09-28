#!/bin/bash

# Build the SFT dataset for Llama 3.2 1B using the shared MegaBlocks generator.

set -euo pipefail

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
MODEL_DATA_DIR="${SCRIPT_DIR}/data"
mkdir -p "${MODEL_DATA_DIR}"

TARGET_DIR="${MODEL_DATA_DIR}" MODEL_ID="${MODEL_ID}" python3 - <<'PY'
import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
target_dir = Path(os.environ["TARGET_DIR"]).resolve()
target_dir.mkdir(parents=True, exist_ok=True)

patterns = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "tokenizer.model",
    "merges.txt",
]

try:
    snapshot_path = snapshot_download(model_id, local_files_only=True, allow_patterns=patterns)
except Exception:
    snapshot_path = snapshot_download(model_id, local_files_only=False, allow_patterns=patterns)

snapshot_path = Path(snapshot_path)
for name in patterns:
    src = snapshot_path / name
    if not src.exists():
        continue
    dst = target_dir / name
    if dst.exists() and dst.samefile(src):
        continue
    shutil.copy2(src, dst)
PY

cd "${SCRIPT_DIR}"
python3 "${REPO_ROOT}/02-generate.sft.shisa-v2.1-megablocks.py" \
    --tokenizer-model "${MODEL_DATA_DIR}" \
    --tokenizer-type HuggingFaceTokenizer \
    --chat-template-model "${MODEL_ID}" \
    "$@"

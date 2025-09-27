#!/bin/bash

# Convert a Megatron-LM checkpoint into a Hugging Face format directory.

set -euo pipefail

usage() {
    cat <<'EOS'
Usage: ./export-hf.sh <run_dir|absolute_path> [--iteration ITER_NAME] [--output OUTPUT_DIR] \
                              [--model-dir MODEL_DIR] [--hf-dtype DTYPE]

Examples:
  # Convert the latest checkpoint from a run created via 03-train-dense.sh
  ./export-hf.sh dense_20250926_194418

  # Convert a specific iteration stored elsewhere
  ./export-hf.sh /workspace/shisa-v2.1/gpt2-125m/checkpoints/dense_20250926_194418 \
      --iteration iter_0002203 --output /workspace/shisa-v2.1/exports/gpt2-125m-hf

Options:
  --iteration   Explicit checkpoint iteration directory name (e.g. iter_0002203)
  --output      Destination directory for Hugging Face files (defaults to <run_dir>-hf)
  --model-dir   Override model directory if tokenizer/data live elsewhere
  --hf-dtype    Desired Hugging Face weight dtype (default: bfloat16)
EOS
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
CWD="$(pwd)"

RUN_ARG=""
ITERATION=""
OUTPUT_ARG=""
MODEL_DIR_OVERRIDE=""
HF_DTYPE="bfloat16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iteration)
            ITERATION="${2:-}"
            shift 2
            ;;
        --output)
            OUTPUT_ARG="${2:-}"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR_OVERRIDE="${2:-}"
            shift 2
            ;;
        --hf-dtype)
            HF_DTYPE="${2:-}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            RUN_ARG="$1"
            shift
            ;;
    esac
done

if [[ -z "${RUN_ARG}" ]]; then
    echo "Error: checkpoint run directory is required." >&2
    usage
    exit 1
fi

resolve_path() {
    local candidate="$1"
    if command -v realpath >/dev/null 2>&1; then
        realpath -m "${candidate}"
    else
        python3 - "${candidate}" <<'PY' 2>/dev/null
import os, sys
print(os.path.realpath(os.path.abspath(os.path.expanduser(sys.argv[1]))))
PY
    fi
}

# Resolve checkpoint directory candidates.
declare -a CANDIDATES=()
if [[ "${RUN_ARG}" = /* ]]; then
    CANDIDATES+=("${RUN_ARG}")
else
    CANDIDATES+=("${RUN_ARG}")
    CANDIDATES+=("${CWD}/${RUN_ARG}")
    CANDIDATES+=("${CWD}/checkpoints/${RUN_ARG}")
    CANDIDATES+=("${SCRIPT_DIR}/${RUN_ARG}")
    CANDIDATES+=("${SCRIPT_DIR}/checkpoints/${RUN_ARG}")
fi

CHECKPOINT_DIR=""
for path in "${CANDIDATES[@]}"; do
    resolved=$(resolve_path "${path}") || continue
    if [[ -d "${resolved}" ]]; then
        CHECKPOINT_DIR="${resolved}"
        break
    fi
done

if [[ -z "${CHECKPOINT_DIR}" ]]; then
    echo "Error: checkpoint directory not found for input '${RUN_ARG}'." >&2
    exit 1
fi

# Determine iteration if not provided.
if [[ -z "${ITERATION}" ]]; then
    if [[ -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
        latest_iter=$(< "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")
        latest_iter=${latest_iter//$'\r'/}
        latest_iter=${latest_iter//$'\n'/}
        if [[ "${latest_iter}" =~ ^[0-9]+$ ]]; then
            ITERATION=$(printf "iter_%07d" "${latest_iter}")
        else
            ITERATION="${latest_iter}"
        fi
    else
        last_path=$(ls -1d "${CHECKPOINT_DIR}"/iter_* 2>/dev/null | sort | tail -n 1 || true)
        if [[ -z "${last_path}" ]]; then
            echo "Error: unable to determine checkpoint iteration (no iter_* directories)." >&2
            exit 1
        fi
        ITERATION="$(basename "${last_path}")"
    fi
fi

if [[ "${ITERATION}" != iter_* ]]; then
    echo "Warning: expected iteration name like iter_0000001, got '${ITERATION}'." >&2
fi

# Determine output directory.
if [[ -z "${OUTPUT_ARG}" ]]; then
    OUTPUT_DIR="${CHECKPOINT_DIR%/}-hf"
else
    if [[ "${OUTPUT_ARG}" = /* ]]; then
        OUTPUT_DIR="${OUTPUT_ARG}"
    else
        OUTPUT_DIR=$(resolve_path "${OUTPUT_ARG}")
    fi
fi
mkdir -p "${OUTPUT_DIR}"

MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/workspace/Megatron-LM}"
if [[ ! -d "${MEGATRON_LM_PATH}" ]]; then
    echo "Error: Megatron-LM directory not found at ${MEGATRON_LM_PATH}." >&2
    exit 1
fi

if [[ -n "${MODEL_DIR_OVERRIDE}" ]]; then
    MODEL_DIR="$(resolve_path "${MODEL_DIR_OVERRIDE}")"
else
    MODEL_DIR=""
    SEARCH_DIR="${CHECKPOINT_DIR}"
    while [[ "${SEARCH_DIR}" != "/" ]]; do
        if [[ -d "${SEARCH_DIR}/data" ]]; then
            MODEL_DIR="${SEARCH_DIR}"
            break
        fi
        NEXT_DIR="$(dirname "${SEARCH_DIR}")"
        [[ "${NEXT_DIR}" == "${SEARCH_DIR}" ]] && break
        SEARCH_DIR="${NEXT_DIR}"
    done
    if [[ -z "${MODEL_DIR}" ]]; then
        MODEL_DIR="$(dirname "${CHECKPOINT_DIR}")"
    fi
fi

MODEL_DATA_DIR="${MODEL_DIR}/data"
if [[ ! -d "${MODEL_DATA_DIR}" ]]; then
    echo "Error: tokenizer data directory not found at ${MODEL_DATA_DIR}. Use --model-dir to specify it explicitly." >&2
    exit 1
fi

LOAD_PATH="${CHECKPOINT_DIR}/${ITERATION}"
if [[ ! -d "${LOAD_PATH}" ]]; then
    LOAD_PATH="${CHECKPOINT_DIR}"
fi

VOCAB_FILE="${MODEL_DATA_DIR}/gpt2-vocab.json"
if [[ ! -f "${VOCAB_FILE}" ]]; then
    VOCAB_FILE=""
fi

MERGES_FILE="${MODEL_DATA_DIR}/gpt2-merges.txt"
if [[ ! -f "${MERGES_FILE}" ]]; then
    MERGES_FILE=""
fi

echo "Converting checkpoint:"
echo "  Source dir: ${CHECKPOINT_DIR}"
[[ "${LOAD_PATH}" != "${CHECKPOINT_DIR}" ]] && echo "  Iter dir  : ${LOAD_PATH}"
echo "  Iteration : ${ITERATION}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Tokenizer : ${MODEL_DATA_DIR}"
echo "  Model dir : ${MODEL_DIR}"

cd "${MEGATRON_LM_PATH}"

TP1_DIR="${OUTPUT_DIR%/}_tp1_pp1"
TP1_ITER_DIR="${TP1_DIR}/${ITERATION}"

if [[ ! -d "${TP1_ITER_DIR}" ]]; then
    mkdir -p "${TP1_DIR}"
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    export MASTER_PORT=${MASTER_PORT:-29500}
    export RANK=0
    export WORLD_SIZE=1
    export WANDB_MODE=${WANDB_MODE:-offline}
    python3 tools/checkpoint/convert.py \
        --model-type GPT \
        --loader mcore \
        --saver megatron \
        --load-dir "${CHECKPOINT_DIR}" \
        --save-dir "${TP1_DIR}" \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --megatron-path "${MEGATRON_LM_PATH}" \
        --loader-transformer-impl transformer_engine
fi

if [[ ! -d "${TP1_ITER_DIR}" ]]; then
    echo "Error: expected converted checkpoint at ${TP1_ITER_DIR} but it was not created." >&2
    exit 1
fi

python3 "${SCRIPT_DIR}/gpt2-125m/convert_dist_megatron_to_hf.py" \
    "${TP1_ITER_DIR}" \
    "${OUTPUT_DIR}" \
    --tokenizer-path "${MODEL_DATA_DIR}"

if [[ -n "${HF_DTYPE}" ]]; then
    python3 - "${OUTPUT_DIR}" "${HF_DTYPE}" <<'PY'
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM

output_dir = Path(sys.argv[1])
dtype_arg = sys.argv[2].lower()

mapping = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "full": torch.float32,
}

if dtype_arg not in mapping:
    supported = ', '.join(sorted(mapping))
    raise SystemExit(f"Unsupported hf-dtype '{dtype_arg}'. Supported: {supported}")

dtype = mapping[dtype_arg]
print(f"Casting Hugging Face weights in {output_dir} to dtype {dtype}...")
model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=dtype, low_cpu_mem_usage=False, trust_remote_code=True)
model.save_pretrained(output_dir, safe_serialization=True)
PY
fi

echo "Conversion complete. Hugging Face checkpoint written to ${OUTPUT_DIR}."


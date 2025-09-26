#!/bin/bash
# MegaBlocks ROCm installation helper

set -euo pipefail

REPO_URL=${MEGABLOCKS_REPO_URL:-https://github.com/ROCm/megablocks.git}
REPO_DIR=${MEGABLOCKS_DIR:-megablocks}

log() {
  echo "[install-megablocks] $*"
}

log "=== Installing MegaBlocks ==="
log "Current working directory: $(pwd)"

if [[ ! -d "${REPO_DIR}" ]]; then
  log "MegaBlocks repository not found; cloning ${REPO_URL} into ${REPO_DIR}"
  git clone --recursive "${REPO_URL}" "${REPO_DIR}"
else
  log "MegaBlocks directory already present (${REPO_DIR}); skipping clone"
fi

if [[ ! -d "${REPO_DIR}" ]]; then
  log "ERROR: megablocks directory still missing after clone attempt"
  exit 1
fi

pushd "${REPO_DIR}" >/dev/null

# Apply ROCm MI300X specific patches idempotently.
python3 - "$PWD" <<'PY'
import pathlib
import re
import sys

repo_path = pathlib.Path(sys.argv[1])
setup_path = repo_path / "setup.py"
hist_path = repo_path / "csrc" / "histogram.h"

# Force targets to gfx942 only
setup_text = setup_path.read_text()
updated_setup_text = re.sub(r"gpus\s*=\s*\[[^\]]*\]", "gpus = ['gfx942']", setup_text, count=1)
if updated_setup_text != setup_text:
    setup_path.write_text(updated_setup_text)

# Ensure histogram uses hipcub alias when hipified
hist_text = hist_path.read_text()
if "megablocks_cub" not in hist_text:
    include_re = re.compile(r"#include <cub/cub\\.cuh>\s*\n", re.MULTILINE)
    replacement = (
        "#if defined(__HIP_PLATFORM_AMD__)\n"
        "#include <hipcub/hipcub.hpp>\n"
        "namespace megablocks_cub = hipcub;\n"
        "#else\n"
        "#include <cub/cub.cuh>\n"
        "namespace megablocks_cub = cub;\n"
        "#endif\n\n"
    )
    hist_text_new = include_re.sub(replacement, hist_text, count=1)
    hist_text_new = hist_text_new.replace("cub::DeviceHistogram::", "megablocks_cub::DeviceHistogram::")
    hist_path.write_text(hist_text_new)
PY

# Clean previous build artifacts to force hipify to regenerate outputs.
log "Cleaning previous MegaBlocks build artifacts"
python3 setup.py clean --all >/dev/null 2>&1 || true
rm -rf build

log "Running setup.py develop --no-deps (for hipify)"
python3 setup.py develop --no-deps

log "Installing MegaBlocks via pip editable (--no-deps)"
python3 -m pip install -e . --no-deps

log "MegaBlocks installation completed successfully"
log "You can validate with: python3 -c 'import megablocks; print(megablocks.__version__)'"

popd >/dev/null
log "Returned to: $(pwd)"

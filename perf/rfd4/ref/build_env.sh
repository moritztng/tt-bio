#!/bin/bash
# Build the upstream RFD4-Proteina CPU fp32 reference environment (row rfd4-ref).
set -euo pipefail
ROOT=/home/moritz/rfd4-ref-env
SHA=2db07bb444d91695f8c10e6acb7648e9297f9221
SRC=$ROOT/src/RFD4-Proteina
export UV_CACHE_DIR=$ROOT/uv-cache

mkdir -p "$ROOT/src"
if [ ! -d "$SRC/.git" ]; then
  git clone /home/moritz/rfd4-upstream/RFD4-Proteina "$SRC"
fi
git -C "$SRC" checkout -q "$SHA"
git -C "$SRC" rev-parse HEAD

cd "$ROOT"
uv venv --python 3.12 "$ROOT/venv"
export VIRTUAL_ENV=$ROOT/venv
uv pip install --python "$ROOT/venv/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  "torch>=2.11,<3"
uv pip install --python "$ROOT/venv/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  "$SRC"
"$ROOT/venv/bin/python" -c "import torch, rfproteina; print('torch', torch.__version__); print('rfproteina ok')"
echo "BUILD_OK"

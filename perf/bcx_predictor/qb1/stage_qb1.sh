#!/bin/bash
# Stage the reference arm on qb1 in tmpfs. qb1's root disk has 1.2 G free; /dev/shm is
# RAM-backed with 252 G. uv's python and cache dirs are redirected here too, or uv would
# write the interpreter to the full disk.
set -u
R=/dev/shm/bcx-ref
LOG=$R/stage.log
mkdir -p "$R"
export UV_PYTHON_INSTALL_DIR=$R/py UV_CACHE_DIR=$R/uvcache
UV=/home/ttuser/.local/bin/uv
{
  echo "=== stage start $(date -u +%FT%TZ) ==="
  rm -rf "$R/venv"
  $UV python install 3.12 && echo "py312 ok"
  $UV venv --python 3.12 "$R/venv" && echo "venv ok"
  $UV pip install --python "$R/venv/bin/python" \
     "jax[cpu]==0.11.2" dm-haiku optax ml_collections jmp numpy scipy pandas biotite 2>&1 | tail -3
  "$R/venv/bin/python" -c "import jax,haiku,optax,jmp,ml_collections,numpy;print('jax',jax.__version__,jax.devices())"
  echo "=== params $(date -u +%FT%TZ) ==="
  mkdir -p "$R/af2_params"; cd "$R/af2_params" || exit 1
  curl -sfL --retry 5 --retry-delay 10 -o p.tar \
    https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
  echo "download rc=$?"
  tar -xf p.tar && rm -f p.tar
  echo "params files: $(ls | wc -l)"
  echo "=== stage done $(date -u +%FT%TZ) ==="
} >>"$LOG" 2>&1

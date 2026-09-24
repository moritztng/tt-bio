#!/bin/bash
set -u
R=/dev/shm/bcx-ref
LOG=$R/params.log
cd "$R/af2_params" || exit 1
{
  echo "=== retry $(date -u +%FT%TZ) ==="
  rm -f p.tar
  curl -fL --retry 5 --retry-delay 10 -o p.tar \
    https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar 2>&1 | tail -1
  echo "curl rc=$? size=$(stat -c %s p.tar 2>/dev/null)"
  tar -xvf p.tar 2>&1 | tail -20
  echo "tar rc=$?"
  rm -f p.tar
  echo "files: $(ls *.npz 2>/dev/null | wc -l)"
  ls -la *.npz 2>/dev/null | head -20
  echo "=== done $(date -u +%FT%TZ) ==="
} >>"$LOG" 2>&1

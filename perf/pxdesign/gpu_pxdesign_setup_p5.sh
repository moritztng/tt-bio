#!/bin/bash
# Phase 5: AF2-IG died in colabdesign's Kabsch SVD with
#   gpusolverDnCreate(&handle) failed: cuSolver internal error
# jax 0.4.29's nvidia-* requirements have no upper bound, so pip resolved cuBLAS 12.9.2.10 /
# cuSolver 11.7.5.82 -- the CUDA 12.9 line, three years newer than the CUDA 12.5 wheels jaxlib
# 0.4.29 shipped against. An isolated svd works; the failure only appears once AF2's jitted graph
# is resident, which is what a library-version skew at handle creation looks like. Pin the wheels
# jax 0.4.29 was released with.
set -u
exec >>/work/setup5.log 2>&1
echo "=== phase5 start $(date -u +%FT%TZ) ==="
PY=$(command -v python3)
$PY -m pip install -q --no-cache-dir \
  "nvidia-cublas-cu12==12.5.3.2" "nvidia-cusolver-cu12==11.6.3.83" \
  "nvidia-cusparse-cu12==12.5.1.3" "nvidia-cufft-cu12==11.2.3.61" \
  "nvidia-cuda-cupti-cu12==12.5.82" "nvidia-cuda-nvcc-cu12==12.5.82" \
  "nvidia-cuda-runtime-cu12==12.5.82" "nvidia-cudnn-cu12==9.2.1.18" \
  "nvidia-nvjitlink-cu12==12.5.82" "numpy==1.26.3" 2>&1 | tail -5
$PY -m pip list 2>/dev/null | grep -i "^nvidia\|^jax"
XLA_PYTHON_CLIENT_PREALLOCATE=false $PY -c "
import jax, jax.numpy as jnp, numpy as np
print('  devices', jax.devices())
a = np.eye(128, dtype=np.float32)
print('  svd', jnp.linalg.svd(jnp.array(a))[1][:2])
import torch; print('  torch still ok', torch.zeros(4, device='cuda').sum().item())
" 2>&1 | tail -5
echo "=== phase5 done $(date -u +%FT%TZ) ==="

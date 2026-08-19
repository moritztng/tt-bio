#!/bin/bash
# Phase 2: the two things phase 1 got wrong.
#  1. `jax[cuda]` resolves to jaxlib 0.4.29+cuda12.cudnn91, a LOCAL-CUDA build that wants system
#     cuDNN 9.1 and cuSPARSE >= 12.1. The image ships cuDNN 8 and cuSPARSE 12.0.2, so JAX silently
#     fell back to CPU -- AF2-IG would have been measured on the CPU. jax[cuda12] pulls the
#     self-contained nvidia-*-cu12 pip wheels instead.
#  2. download_inference_cache only fetches the checkpoint for configs.model_name. The three
#     Protenix eval checkpoints (base, mini, mini_tmpl) are fetched lazily inside the eval stage.
set -u
exec >>/work/setup2.log 2>&1
echo "=== phase2 start $(date -u +%FT%TZ) ==="
FAIL() { echo "PHASE2 FAILED: $*"; echo "$*" > /work/SETUP_FAIL; exit 1; }
PY=$(command -v python3)

$PY -m pip uninstall -y -q jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt 2>/dev/null
$PY -m pip install -q --no-cache-dir "jax[cuda12]==0.4.29" || FAIL "jax[cuda12]"
$PY -m pip install -q --no-cache-dir "numpy==1.26.3" || FAIL "numpy repin"

echo "--- jax devices"
XLA_PYTHON_CLIENT_PREALLOCATE=false $PY -c "
import jax, jaxlib
print('  jax', jax.__version__, 'jaxlib', jaxlib.__version__)
d = jax.devices()
print('  devices', d)
assert d[0].platform == 'gpu', 'JAX STILL ON CPU'
import jax.numpy as jnp
x = jnp.ones((2048, 2048)); print('  matmul ok', float((x @ x).sum()))
" || FAIL "jax gpu"

CKPT=/work/PXDesign/release_data/checkpoint
mkdir -p $CKPT
for m in pxdesign_v0.1.0 protenix_base_default_v0.5.0 protenix_mini_default_v0.5.0 \
         protenix_mini_tmpl_v0.5.0; do
  U=https://pxdesign.tos-cn-beijing.volces.com/release_model/$m.pt
  if [ ! -s $CKPT/$m.pt ] || [ $(stat -c%s $CKPT/$m.pt) -lt 10000000 ]; then
    echo "--- fetching $m"
    curl -fsSL -o $CKPT/$m.pt.part "$U" || FAIL "download $m"
    mv $CKPT/$m.pt.part $CKPT/$m.pt
  fi
  $PY -c "
import torch, sys
ck = torch.load('$CKPT/$m.pt', map_location='cpu')
n = len(ck.get('model', ck)) if isinstance(ck, dict) else -1
print('  $m ok, %d top-level entries' % (len(ck) if isinstance(ck, dict) else -1))
" || FAIL "verify $m"
done
ls -la $CKPT
sha256sum $CKPT/*.pt

echo "=== phase2 ok $(date -u +%FT%TZ) ==="
echo ok > /work/SETUP2_OK

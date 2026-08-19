#!/bin/bash
# Phase 3: kill phase 2's serial curl of the Protenix checkpoints (the volces cn-beijing origin
# served them at ~20 KB/s from New Jersey, which is 14 h for a 1 GB file) and refetch with 16
# parallel range connections. Also the two dependency holes phase 2 opened: jax[cuda12] dragged
# numpy to 2.2.6 (protenix pins 1.26.3) and colabdesign needs immutabledict, which install.sh's
# --no-deps install never provided.
set -u
exec >>/work/setup3.log 2>&1
echo "=== phase3 start $(date -u +%FT%TZ) ==="
FAIL() { echo "PHASE3 FAILED: $*"; echo "$*" > /work/SETUP_FAIL; exit 1; }
PY=$(command -v python3)

pkill -f 'curl.*release_model' 2>/dev/null
pkill -f fix2.sh 2>/dev/null
sleep 1

$PY -m pip install -q --no-cache-dir "numpy==1.26.3" immutabledict || FAIL "numpy/immutabledict"
apt-get install -y -qq aria2 >/dev/null 2>&1 || FAIL "aria2"

echo "--- jax devices after numpy repin"
XLA_PYTHON_CLIENT_PREALLOCATE=false $PY -c "
import jax, jaxlib, numpy
print('  jax', jax.__version__, 'jaxlib', jaxlib.__version__, 'numpy', numpy.__version__)
d = jax.devices(); print('  devices', d)
assert d[0].platform == 'gpu', 'JAX STILL ON CPU'
import jax.numpy as jnp
x = jnp.ones((2048, 2048)); print('  matmul ok', float((x @ x).sum()))
" || FAIL "jax gpu"

CKPT=/work/PXDesign/release_data/checkpoint
mkdir -p $CKPT
B=https://pxdesign.tos-cn-beijing.volces.com/release_model
for m in protenix_base_default_v0.5.0 protenix_mini_default_v0.5.0 protenix_mini_tmpl_v0.5.0; do
  rm -f $CKPT/$m.pt $CKPT/$m.pt.part
  echo "--- size check $m: $(curl -sSI $B/$m.pt | grep -i content-length)"
  aria2c -x16 -s16 -k1M --file-allocation=none --summary-interval=30 \
         -d $CKPT -o $m.pt "$B/$m.pt" || FAIL "aria2 $m"
done
for m in pxdesign_v0.1.0 protenix_base_default_v0.5.0 protenix_mini_default_v0.5.0 \
         protenix_mini_tmpl_v0.5.0; do
  $PY -c "
import torch
ck = torch.load('$CKPT/$m.pt', map_location='cpu')
print('  $m loads, %d top-level entries' % (len(ck) if isinstance(ck, dict) else -1))
" || FAIL "verify $m"
done
ls -la $CKPT
sha256sum $CKPT/*.pt
echo "=== phase3 ok $(date -u +%FT%TZ) ==="
echo ok > /work/SETUP2_OK

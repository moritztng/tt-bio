#!/bin/bash
# Phase 4: protenix asks for `deepspeed>=0.15.1` with no upper bound, so pip installed 0.19.5,
# whose compile/custom_ops module calls torch.library.custom_op -- added in torch 2.4, and PXDesign
# pins torch 2.3.1. The import blows up inside protenix's primitives, i.e. before the DeepSpeed Evo
# kernel could ever be reached. Pin deepspeed to the 0.15 line the requirement floor names.
set -u
exec >>/work/setup4.log 2>&1
echo "=== phase4 start $(date -u +%FT%TZ) ==="
FAIL() { echo "PHASE4 FAILED: $*"; echo "$*" > /work/SETUP_FAIL; exit 1; }
PY=$(command -v python3)
for V in 0.15.4 0.15.1 0.16.5; do
  echo "--- trying deepspeed==$V"
  $PY -m pip install -q --no-cache-dir "deepspeed==$V" >/dev/null 2>&1 || continue
  $PY -m pip install -q --no-cache-dir "numpy==1.26.3" >/dev/null 2>&1
  if $PY -c "
import importlib.util
assert importlib.util.find_spec('deepspeed.ops.deepspeed4science') is not None
from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention
from protenix.openfold_local.model.primitives import LayerNorm, DS4Sci_EvoformerAttention as D2
import deepspeed, torch
print('  deepspeed', deepspeed.__version__, 'torch', torch.__version__, 'evo import OK')
"; then echo "PICKED deepspeed==$V"; echo "$V" > /work/DEEPSPEED_VER; break; fi
done
[ -s /work/DEEPSPEED_VER ] || FAIL "no deepspeed version imports with torch 2.3.1"
XLA_PYTHON_CLIENT_PREALLOCATE=false $PY -c "
import jax, numpy; d=jax.devices(); print('  jax', jax.__version__, d, 'numpy', numpy.__version__)
assert d[0].platform=='gpu'
import pxdesign, pxdbench, protenix, colabdesign; print('  all imports OK')
" || FAIL "post-pin import check"
echo "=== phase4 ok $(date -u +%FT%TZ) ==="
echo ok > /work/SETUP4_OK

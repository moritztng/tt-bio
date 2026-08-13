#!/bin/bash
# Run every RFD3 arm on one rented box, in the order that keeps the arms honest.
#
#   $1 = gpu label (H200 | B200)   $2 = measured power.limit W   $3 = measured idle W
#   $4 = "skipD" to drop the sparse control (B200, budget)
#
# Order matters. cuequivariance is installed LAST, because arm C's whole point is that arms A, B and
# D ran with it absent. Arm A also runs first so its atom count becomes --ref-atoms for every later
# arm: a silently different structure then cannot pass as a fast one.
#
# Smoke first, 4 timesteps: it catches the `length` trap, a missing C compiler and the weight
# download in 30 s instead of inside a 6-minute measured point.
set -u
GPU=${1:?gpu label}; PLIM=${2:?power limit W}; IDLE=${3:?idle W}; SKIPD=${4:-}
exec >>/work/runall.log 2>&1
echo "=== runall $GPU start $(date -u +%FT%TZ)  plim=$PLIM idle=$IDLE ==="
cd /work || exit 1
JL=/work/results/rfd3_prod.jsonl
FIX=perf/dsfix/fixtures/rfd3_R4_gpu.json

P() { /work/v_head/bin/python perf/dsfix/gpu_rfd3_prod.py --gpu "$GPU" \
      --power-limit "$PLIM" --idle-W "$IDLE" --inputs "$FIX" --out "$JL" "$@"; }

echo "--- smoke (v_head, 4 timesteps, b=1) $(date -u +%FT%TZ)"
rm -rf /work/out/smoke
/work/v_head/bin/python perf/dsfix/gpu_rfd3_run.py --counts /work/out/smoke.json -- \
  design out_dir=/work/out/smoke inputs="$FIX" inference_sampler.num_timesteps=4 \
  diffusion_batch_size=1 n_batches=1 seed=7 skip_existing=False
echo "--- smoke counts:"; cat /work/out/smoke.json
ls -la /work/out/smoke | head
if ! grep -q '"dense_sdpa_pairbias_attention": [1-9]' /work/out/smoke.json; then
  echo "SMOKE FAILED: dense path did not run on arm A"; echo smoke > /work/RUN_FAIL; exit 1
fi

echo "--- arm A head-fast b=8,1 $(date -u +%FT%TZ)"
P --arm head-fast --runner /work/v_head/bin/python --batches 8 1

REF=$(python3 -c "
import json
a=[json.loads(l) for l in open('$JL')]
a=[r for r in a if r['arm']=='head-fast' and r['valid']]
print(a[0]['evidence']['atoms'] if a else '')
")
if [ -z "$REF" ]; then echo "no valid arm A point, refusing to run the rest"; echo armA > /work/RUN_FAIL; exit 1; fi
echo "--- ref atoms = $REF"

if [ "$SKIPD" != "skipD" ]; then
  echo "--- arm D head-sparse b=8 $(date -u +%FT%TZ)"
  RFD3_DENSE_SDPA_ATTENTION=0 P --arm head-sparse --runner /work/v_head/bin/python \
    --batches 8 --ref-atoms "$REF"
fi

echo "--- arm B pip-0.2.0 b=8 $(date -u +%FT%TZ)"
P --arm pip-0.2.0 --runner /work/v_pip/bin/python --batches 8 --ref-atoms "$REF"

echo "--- installing cuequivariance-torch for arm C $(date -u +%FT%TZ)"
/opt/conda/bin/uv pip install -q --python /work/v_head/bin/python cuequivariance-torch \
  || echo "cueq install failed"
/work/v_head/bin/python -c "
from importlib.metadata import version, PackageNotFoundError
for p in ('cuequivariance-torch','cuequivariance-ops-torch','cuequivariance'):
    try: print(p, version(p))
    except PackageNotFoundError: print(p, None)
"
echo "--- arm C head-cueq b=8 $(date -u +%FT%TZ)"
P --arm head-cueq --runner /work/v_head/bin/python --batches 8 --ref-atoms "$REF"

echo "=== runall $GPU done $(date -u +%FT%TZ) ==="
echo ok > /work/RUN_OK

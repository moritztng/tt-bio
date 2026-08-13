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

# The spec's `input` is relative and upstream resolves it against the SPEC FILE's own directory, not
# cwd -- from perf/dsfix/fixtures/ it went looking for
# /work/perf/dsfix/fixtures/perf/dsfix/targets/R4_9q6y_A.pdb and died in the load_input validator.
# So the spec is copied to /work, where its relative target path resolves. Same bytes, same sha256
# (647e066a...), just parked where the resolver expects it.
FIX=/work/rfd3_R4_gpu.json
cp -f /work/perf/dsfix/fixtures/rfd3_R4_gpu.json "$FIX"
sha256sum "$FIX"

P() { /work/v_head/bin/python perf/dsfix/gpu_rfd3_prod.py --gpu "$GPU" \
      --power-limit "$PLIM" --idle-W "$IDLE" --inputs "$FIX" --out "$JL" "$@"; }

# A point already measured and validated is never re-paid for: rental time is the budget here.
HAVE() { python3 -c "
import json,os,sys
p='$JL'
if not os.path.exists(p): sys.exit(1)
for l in open(p):
    r=json.loads(l)
    if r['arm']=='$1' and r['batch']==int('$2') and r['valid']: sys.exit(0)
sys.exit(1)
"; }

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

if HAVE head-fast 8 && HAVE head-fast 1; then
  echo "--- arm A already measured and valid, skipping"
else
  echo "--- arm A head-fast b=8,1 $(date -u +%FT%TZ)"
  P --arm head-fast --runner /work/v_head/bin/python --batches 8 1
fi

# Arm A's residue topology becomes the reference for every later arm. Atom counts vary per design
# (new sequence each time), the topology does not, so this is the invariant that catches an arm
# that quietly designed something else.
REF=$(python3 -c "
import json
a=[json.loads(l) for l in open('$JL')]
a=[r for r in a if r['arm']=='head-fast' and r['valid']]
print(json.dumps(a[0]['evidence']['residues_per_chain']) if a else '')
")
if [ -z "$REF" ]; then echo "no valid arm A point, refusing to run the rest"; echo armA > /work/RUN_FAIL; exit 1; fi
echo "--- ref chains = $REF"

if [ "$SKIPD" != "skipD" ]; then
  if HAVE head-sparse 8; then echo "--- arm D already measured, skipping"; else
    echo "--- arm D head-sparse b=8 $(date -u +%FT%TZ)"
    RFD3_DENSE_SDPA_ATTENTION=0 P --arm head-sparse --runner /work/v_head/bin/python \
      --batches 8 --ref-chains "$REF"
  fi
fi

if HAVE pip-0.2.0 8; then echo "--- arm B already measured, skipping"; else
  echo "--- arm B pip-0.2.0 b=8 $(date -u +%FT%TZ)"
  P --arm pip-0.2.0 --runner /work/v_pip/bin/python --batches 8 --ref-chains "$REF"
fi

echo "--- installing cuequivariance-torch for arm C $(date -u +%FT%TZ)"
/opt/conda/bin/uv pip install -q --python /work/v_head/bin/python cuequivariance-torch \
  || echo "cueq install failed"
/work/v_head/bin/python -c "
from importlib.metadata import version, PackageNotFoundError
for p in ('cuequivariance-torch','cuequivariance-ops-torch','cuequivariance'):
    try: print(p, version(p))
    except PackageNotFoundError: print(p, None)
"
if HAVE head-cueq 8; then echo "--- arm C already measured, skipping"; else
  echo "--- arm C head-cueq b=8 $(date -u +%FT%TZ)"
  P --arm head-cueq --runner /work/v_head/bin/python --batches 8 --ref-chains "$REF"
fi

echo "=== runall $GPU done $(date -u +%FT%TZ) ==="
echo ok > /work/RUN_OK

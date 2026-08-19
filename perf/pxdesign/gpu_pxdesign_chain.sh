#!/bin/bash
# Everything after the anchor sweep, chained so the card is never idle and never shared. Serial on
# purpose: two concurrent cells on one card would make every number a co-tenancy artefact.
set -u
exec >>/work/chain.log 2>&1
cd /work/PXDesign
export TOOL_WEIGHTS_ROOT=/work/tool_weights
export PROTENIX_DATA_ROOT_DIR=/work/release_data/ccd_cache
export CUTLASS_PATH=/work/cutlass
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
R=/work/results
S="python3 -u /work/gpu_pxdesign_sweep.py"

echo "=== chain start $(date -u +%FT%TZ) ==="
while pgrep -f "gpu_pxdesign_sweep.py --cells anchor" >/dev/null; do sleep 20; done
echo "--- anchor sweep exited $(date -u +%FT%TZ)"

# --- MSA prep for the ladder, OUTSIDE any timed cell -------------------------------------------
# The ptx filter calls populate_msa_with_cache, which shells out to `protenix msa` (an online
# search) for any target chain without a precomputed MSA. Left alone that runs inside the timed
# Protenix stage and makes the cell a measurement of somebody else's web service. Warm it here and
# record whether it worked; the ladder's extended arm is only valid if it did.
for f in /work/targets/lacz_*.yaml; do
  echo "--- prepare-msa $f $(date -u +%FT%TZ)"
  timeout 1800 pxdesign prepare-msa --yaml "$f" && echo "  MSA OK $f" || echo "  MSA FAILED $f"
done
grep -l "msa:" /work/targets/lacz_*.yaml > /work/MSA_READY 2>/dev/null || true
cat /work/targets/lacz_128.yaml

LAD="lacz128:/work/targets/lacz_128.yaml,lacz256:/work/targets/lacz_256.yaml"
LAD="$LAD,lacz512:/work/targets/lacz_512.yaml,lacz768:/work/targets/lacz_768.yaml"
mk() { local pre=$1 n=$2; local o=""; for c in ${LAD//,/ }; do o="$o,${c%%:*}_${pre:0:4}_n${n}:${c#*:}:${pre}:${n}"; done; echo "${o:1}"; }

echo "=== LEG 1: ladder, preview, N=1 (no Protenix, so no MSA dependency) ==="
$S --cells "$(mk preview 1)" --reps 2 --results $R/pxd_ladder.jsonl

echo "=== LEG 2: ladder, extended, N=1 ==="
$S --cells "$(mk extended 1)" --reps 2 --results $R/pxd_ladder.jsonl

echo "=== LEG 3: the MSA axis on the anchor -- same cell, ptx MSA off ==="
$S --cells "pdl1:examples/PDL1_quick_start.yaml:extended:1" --reps 2 \
   --label-suffix "_ext_n1_nomsa" --extra "--eval.binder.tools.ptx.use_msa false" \
   --results $R/pxd_msaaxis.jsonl

echo "=== LEG 4: saturation probe -- can anything actually load an H200 ==="
$S --cells "lacz768:/work/targets/lacz_768.yaml:extended:8" --reps 1 \
   --label-suffix "_ext_n8" --results $R/pxd_sat.jsonl
$S --cells "lacz128:/work/targets/lacz_128.yaml:extended:8" --reps 1 \
   --label-suffix "_ext_n8" --results $R/pxd_sat.jsonl

echo "=== chain done $(date -u +%FT%TZ) ==="
echo done > /work/CHAIN_DONE

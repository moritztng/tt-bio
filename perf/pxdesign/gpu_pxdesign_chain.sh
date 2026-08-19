#!/bin/bash
# Ladder + MSA axis + saturation, relaunched after the ladder MSAs were sliced to their crops.
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
LAD="lacz128:/work/targets/lacz_128.yaml,lacz256:/work/targets/lacz_256.yaml"
LAD="$LAD,lacz512:/work/targets/lacz_512.yaml,lacz768:/work/targets/lacz_768.yaml"
mk() { local pre=$1 n=$2; local o=""; for c in ${LAD//,/ }; do o="$o,${c%%:*}_${pre:0:4}_n${n}:${c#*:}:${pre}:${n}"; done; echo "${o:1}"; }

echo "=== chain3 start $(date -u +%FT%TZ) ==="
echo "=== LEG 1: ladder, preview, N=1 ==="
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
echo "=== chain3 done $(date -u +%FT%TZ) ==="
echo done > /work/CHAIN_DONE

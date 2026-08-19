#!/bin/bash
# Final measurement chain, on the per-rung cropped CIFs (targets2).
#
# The ladder's Protenix arm runs with --eval.binder.tools.ptx.use_msa false. Not a shortcut: after
# `crop` turned out to leave the entity sequence uncropped, the only MSA a cropped rung can have is
# a fresh search of the cropped sequence, and that search is an external service that took >4 min
# on the first rung with no bound. Running the ladder MSA-free keeps the size axis clean and keeps
# an outside web service out of a timed stage. `use_msa false` is a configuration the pipeline
# itself selects in its target-template branch, not an invented one. LEG 3 prices what the MSA adds
# by measuring the anchor both ways, where a real published MSA exists.
set -u
exec >>/work/chain4.log 2>&1
cd /work/PXDesign
export TOOL_WEIGHTS_ROOT=/work/tool_weights
export PROTENIX_DATA_ROOT_DIR=/work/release_data/ccd_cache
export CUTLASS_PATH=/work/cutlass
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
R=/work/results
S="python3 -u /work/gpu_pxdesign_sweep.py"
T=/work/targets2
NOMSA="--eval.binder.tools.ptx.use_msa false"
mk() { local pre=$1 n=$2 o=""; for s in 128 256 512 768; do
         o="$o,laczc${s}_${pre:0:4}_n${n}:$T/laczc_${s}.yaml:${pre}:${n}"; done; echo "${o:1}"; }

echo "=== chain4 start $(date -u +%FT%TZ) ==="
echo "=== LEG 1: ladder, preview, N=1 ==="
$S --cells "$(mk preview 1)" --reps 2 --results $R/pxd_ladder2.jsonl
echo "=== LEG 2: ladder, extended, N=1, ptx MSA off ==="
$S --cells "$(mk extended 1)" --reps 2 --extra "$NOMSA" --results $R/pxd_ladder2.jsonl
echo "=== LEG 3: MSA axis on the anchor -- same cell, ptx MSA off ==="
$S --cells "pdl1:examples/PDL1_quick_start.yaml:extended:1" --reps 2 \
   --label-suffix "_ext_n1_nomsa" --extra "$NOMSA" --results $R/pxd_msaaxis.jsonl
echo "=== LEG 4: saturation -- can anything load an H200 ==="
$S --cells "laczc768:$T/laczc_768.yaml:extended:8" --reps 1 --label-suffix "_ext_n8" \
   --extra "$NOMSA" --results $R/pxd_sat.jsonl
$S --cells "laczc128:$T/laczc_128.yaml:extended:8" --reps 1 --label-suffix "_ext_n8" \
   --extra "$NOMSA" --results $R/pxd_sat.jsonl
echo "=== LEG 5: fast LayerNorm actually reached ==="
LAYERNORM_TYPE=fast_layernorm $S --cells "pdl1:examples/PDL1_quick_start.yaml:extended:1" \
   --reps 2 --label-suffix "_ext_n1_fastln" --results $R/pxd_fastln.jsonl
LAYERNORM_TYPE=fast_layernorm $S --cells "laczc512:$T/laczc_512.yaml:preview:1" \
   --reps 2 --label-suffix "_prev_n1_fastln" --results $R/pxd_fastln.jsonl

echo "=== build the reference json $(date -u +%FT%TZ) ==="
python3 /work/make_reference.py \
  --jsonl $R/pxd_gpu.jsonl $R/pxd_ladder2.jsonl $R/pxd_msaaxis.jsonl $R/pxd_sat.jsonl \
          $R/pxd_fastln.jsonl \
  --manifest /work/targets2/manifest.json --out $R/gpu_reference.json
echo "=== chain4 done $(date -u +%FT%TZ) ==="
echo done > /work/ALL_DONE

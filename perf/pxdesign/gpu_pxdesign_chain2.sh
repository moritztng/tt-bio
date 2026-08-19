#!/bin/bash
# Leg 5, chained after the main chain: price the fast-LayerNorm bug.
#
# protenix's primitives module snapshots LAYERNORM_TYPE into a module-level constant at IMPORT
# time. pxdesign sets that variable in configure_runtime_env, which runs long after the import
# chain has already pulled protenix in, so `--use_fast_ln True` never reaches FusedLayerNorm --
# proven on CPU: the factory returns OpenFoldLayerNorm with the flag on, and FusedLayerNorm only
# when the variable is exported before python starts. Every cell above is therefore a
# NON-fused-LayerNorm baseline. This leg measures the same cell with the variable exported, which
# is what upstream intended, so the two arms bracket the real H200 reference.
set -u
exec >>/work/chain2.log 2>&1
cd /work/PXDesign
export TOOL_WEIGHTS_ROOT=/work/tool_weights
export PROTENIX_DATA_ROOT_DIR=/work/release_data/ccd_cache
export CUTLASS_PATH=/work/cutlass
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
R=/work/results

echo "=== chain2 waiting for chain $(date -u +%FT%TZ) ==="
while [ ! -s /work/CHAIN_DONE ]; do sleep 30; done
echo "=== LEG 5: fast LayerNorm actually reached $(date -u +%FT%TZ) ==="
LAYERNORM_TYPE=fast_layernorm python3 -u /work/gpu_pxdesign_sweep.py \
  --cells "pdl1:examples/PDL1_quick_start.yaml:extended:1" --reps 2 \
  --label-suffix "_ext_n1_fastln" --results $R/pxd_fastln.jsonl
LAYERNORM_TYPE=fast_layernorm python3 -u /work/gpu_pxdesign_sweep.py \
  --cells "lacz512:/work/targets/lacz_512.yaml:preview:1" --reps 2 \
  --label-suffix "_prev_n1_fastln" --results $R/pxd_fastln.jsonl

echo "=== build the reference json $(date -u +%FT%TZ) ==="
python3 /work/make_reference.py \
  --jsonl $R/pxd_gpu.jsonl $R/pxd_ladder.jsonl $R/pxd_msaaxis.jsonl $R/pxd_sat.jsonl \
          $R/pxd_fastln.jsonl \
  --manifest /work/targets/manifest.json --out $R/gpu_reference.json
echo "=== chain2 done $(date -u +%FT%TZ) ==="
echo done > /work/ALL_DONE

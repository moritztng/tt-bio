#!/bin/bash
# PXDesign's batching term at 512 residues: the size the perf page publishes and the one rung the
# GPU reference never ran at N_sample 8. Amortisation is strongly size-dependent (5.25x at 116
# residues, 1.43x at 768), so it cannot be interpolated.
#
# Protocol is the reference's own, unchanged except for --N_sample: preview preset, 400 steps,
# bf16, seed 42, the same laczc_512 yaml/CIF, rep 0 discarded cold, three reps so two are warm.
# Eval stays ON because the harness validates on the output and a run with eval off returns
# ok=False ("no design_outputs summary.csv - ranking never ran"); an invalid run is not a cell.
# The N=1 arm is the control that has to reproduce the published 30.8129 s generator stage on this
# box before the N=8 number counts.
set -u
exec >>/work/chain_final.log 2>&1
cd /work/PXDesign
export TOOL_WEIGHTS_ROOT=/work/tool_weights
export PROTENIX_DATA_ROOT_DIR=/work/release_data/ccd_cache
export CUTLASS_PATH=/work/cutlass
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
R=/work/results; T=/work/targets2
S="python3 -u /work/gpu_pxdesign_sweep.py"
echo "=== chain_final start $(date -u +%FT%TZ) ==="
for arm in 1 8; do
  echo "=== N_sample $arm, 3 reps $(date -u +%FT%TZ) ==="
  $S --cells "laczc512:$T/laczc_512.yaml:preview:$arm" --reps 3 \
     --label-suffix "_prev_n${arm}_b2" --results $R/pxd_final.jsonl
  echo "=== N_sample $arm done $(date -u +%FT%TZ) ==="
done
echo "=== chain_final done $(date -u +%FT%TZ) ==="
echo done > /work/FINAL_DONE

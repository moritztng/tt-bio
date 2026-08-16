#!/usr/bin/env bash
# Step 2: the size sweep for BD, one process per size (one device context per process).
# 298 FIRST: it is the only non-tile-aligned size, so it is the one that exercises B's
# `ttnn.pad` path and D's `padded_activation` refusal. Then 640/768, the sizes where an
# L1-tuned gate has gone dark before (memory tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa).
#
# Per size the acceptance is in the JSON, not in the wall clock:
#   * BD's cif sha + plddt identical to ship's at that size  (bit-exactness, non-negotiable)
#   * lm_handoff[0] > 0 in the BD arm                        (B fired, the arm is not vacuous)
#   * dual_noc[0] > 0, or dual_noc_rejects keyed padded_activation (D fired or declined by design)
#   * BD median <= ship median + that size's ship spread     (no regression)
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-p2
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200-p2
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_WAIT_S=5400
for SZ in 298 640 768 256 512; do
  echo "=== size $SZ ==="
  /home/ttuser/.coworker/scripts/benchlock.sh esmfold2-beat-dgx-h200-p2 -- \
    /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/esm3p4land/fold_ab.py \
    --model esmfold2 --size "$SZ" --rounds 3 --arms ship,BD \
    --out "perf/esmbeat/p2_sweep_${SZ}_c0.json"
  echo "EXIT_$SZ=$?"
done

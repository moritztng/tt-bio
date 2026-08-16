#!/usr/bin/env bash
# Lever E's size sweep. 768 and 640 FIRST: those are the sizes where an L1-tuned gate on this exact
# model has gone dark before (memory tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa), and E adds
# L1 pressure the shipped path does not have. Then 320 (the window's lower edge, where E first
# serves) and 298 (just below it, where E must be inert and non-tile-aligned).
#
# Acceptance per size is in the JSON, not the wall clock:
#   * one cif sha and one plddt across both arms       (bit-exactness, non-negotiable)
#   * l1_ln_stats[0] > 0 in `ship` inside the window   (the arm executed, not vacuous)
#   * l1_ln_stats == [0, 0] in both arms below it      (the gate correctly does not reach)
# rounds=1 because the acceptance is the hash, not the seconds. qb1 is not the page host.
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-p3
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200-p3
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_WAIT_S=5400
for SZ in "$@"; do
  echo "=== size $SZ ==="
  /home/ttuser/.coworker/scripts/benchlock.sh esmfold2-beat-dgx-h200-p3 -- \
    /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/esm3p4land/fold_ab.py \
    --model esmfold2 --size "$SZ" --rounds 1 --arms ship,noE \
    --out "perf/esmbeat/p3_e_sweep_${SZ}_c0.json"
  echo "EXIT_$SZ=$?"
done

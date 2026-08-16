#!/usr/bin/env bash
# Lever G's fold-level A/B, `ship` vs `noG`, on CARD 2. Card 0 is running the release gate, so
# this runs beside it rather than behind it; the acceptance here is a hash comparison, not a wall
# clock, which is exactly what makes that safe (the seconds in these legs are single-shot and
# under a loaded box, so they are directional at best).
#
# Acceptance per size, in the JSON:
#   * one cif sha256 and one plddt across both arms          (bit-exactness, non-negotiable)
#   * fill_assembly_stats[0] > 0 in `ship` inside the window (the arm executed, not vacuous)
#   * fill_assembly_stats == [0, n] in `noG`                 (the control really has it off)
# 512 first because it is the size the ledger is written in, then 1024 (top of the window, the
# only place a refusal is plausible), then 320 (the window's lower edge) and 256 (below it, where
# G must never be asked).
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-p3
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200-p3
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
for SZ in "$@"; do
  echo "=== size $SZ === $(date -u +%FT%TZ)"
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/esm3p4land/fold_ab.py \
    --model esmfold2 --size "$SZ" --rounds 1 --arms ship,noG \
    --out "perf/esmbeat/p3_g_sweep_${SZ}_c2.json"
  echo "EXIT_$SZ=$? $(date -u +%FT%TZ)"
done
echo "G_SWEEP_DONE $(date -u +%FT%TZ)"

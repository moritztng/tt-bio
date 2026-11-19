#!/bin/bash
# abab.sh OUT ROUNDS ARM...  : RFD4 cell (design_cell.py --model rfd4 --fast, 336 tokens) per lever arm, ARM order
# repeated ROUNDS times on one chip, so every arm sees the same chip and clock drift. Run under run_cell.sh (it sets
# PYTHONPATH, TT_VISIBLE_DEVICES and samples AICLK). An arm is `base` (the shipped set) or `+a,+b` added to it.
# Then: python perf/rfd4/spdm/abab_read.py OUT
set -u
OUT=$1 R=$2; shift 2
BASE=$(python -c "from tt_bio.rfd4.trunk import FAST_LEVERS as F; print(','.join(sorted(F)))") || exit 2
for r in $(seq 1 "$R"); do
  for arm in "$@"; do
    L=$BASE; [ "$arm" = base ] || L="$BASE,${arm//+/}"
    echo "$(date -u +%FT%TZ) round $r arm $arm" >&2
    TT_BIO_LEVERS=$L python perf/bmk/design_cell.py --model rfd4 --fast --repeat 3 \
      --out "$OUT/$arm.r$r.json" || echo "arm $arm round $r rc=$?" >&2
  done
done

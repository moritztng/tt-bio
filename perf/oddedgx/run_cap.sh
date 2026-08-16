#!/usr/bin/env bash
# Run 5: capacity. The host glue must not move a digest at any size the model ships at, and
# 640 aa is the size that binds (the divisor-group search drops to g=6 above it).
# Acceptance: per size, the noglue and glue CIF digests are identical and plDDT matches to 6 dp.
set -u
source /home/ttuser/.coworker/wt/opendde-beat-dgx-h200/perf/oddedgx/env.sh
cd $WT || exit 1
echo "=== capacity 256/298/640, noglue vs glue, card 2 $(date -Is) ==="
$BL opendde-beat-dgx-h200 -- $PY -u perf/size512/fold_ab512.py --model opendde \
    --sizes 256,298,640 --arms noglue,glue --out $O/cap_ab_c2.json
echo "RC=$? $(date -Is)"

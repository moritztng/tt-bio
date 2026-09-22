#!/usr/bin/env bash
# of3t-covdefault: the whole inference A/B matrix, one process per fold, card 2 (UMD logical),
# alternating arms so a drift in the box cannot land on one arm.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-covdefault
cd "$W"
L=/tmp/of3t/of3t-covdefault/matrix.log
: > "$L"
for rep in a b c; do
  for m in openfold3 openbind; do
    p=of3; [ "$m" = openbind ] && p=ob
    for arm in off on; do
      bash perf/of3t_covdefault/foldarm.sh "$arm" 0 2 "${p}_${arm}_s0_${rep}" "$m" >> "$L" 2>&1
      echo "done ${p}_${arm}_s0_${rep} rc=$?" >> "$L"
    done
  done
done
# the seed floor: same arm, a different seed
for m in openfold3 openbind; do
  p=of3; [ "$m" = openbind ] && p=ob
  bash perf/of3t_covdefault/foldarm.sh off 1 2 "${p}_off_s1_a" "$m" >> "$L" 2>&1
done
echo "MATRIX COMPLETE $(date -u +%FT%TZ)" >> "$L"

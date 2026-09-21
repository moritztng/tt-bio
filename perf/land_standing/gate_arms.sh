#!/bin/bash
# Release gate, arm by arm, at the wk/land-standing tip — discipline item 2 for D56.
# size-ladder is deliberately absent: its openfold3 p150a check is already running on card 0
# and the two would collide in perf/sizegate/work/.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/land_standing/out/gate_d56
mkdir -p "$OUT"
cd "$WT" || exit 1

ARMS="boltz2 esmfold2 esmfold2-fast protenix-v2 protenix-v1 opendde openfold3 rf3 openbind
      boltzgen rfd3 opendde-abag capacity l1-budget batch-position nesso1 pxdesign
      rf3-1024aa rfd3-fusion esmc-300m esmc-600m"

{
  echo "release gate arm by arm, branch tip $(git rev-parse --short HEAD), started $(date -u +%FT%TZ)"
  echo "card 2, size-ladder excluded (running separately on card 0)"
} >> "$OUT/SUMMARY.txt"

for arm in $ARMS; do
  [ -f "$OUT/done_$arm" ] && continue
  t0=$(date +%s)
  TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:land-standing \
  PYTHONPATH=$WT RELEASE_GATE_CENSUS_PYTHONPATH=$WT \
  "$PY" scripts/release_gate.py --model "$arm" > "$OUT/$arm.log" 2>&1
  rc=$?
  t1=$(date +%s)
  verdict=$(grep -Eo '\b(PASS|FAIL|SKIP)\b' "$OUT/$arm.log" | tail -1)
  echo "$arm rc=$rc verdict=${verdict:-none} wall=$((t1-t0))s $(date -u +%FT%TZ)" >> "$OUT/SUMMARY.txt"
  touch "$OUT/done_$arm"
done
echo "ALL ARMS ATTEMPTED $(date -u +%FT%TZ)" >> "$OUT/SUMMARY.txt"

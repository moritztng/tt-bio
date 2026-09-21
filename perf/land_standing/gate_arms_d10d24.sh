#!/bin/bash
# D10+D24 release gate, arm by arm.
#
# Two things this script got wrong the first time and now handles:
#  - release_gate.py has a PREFLIGHT LOAD CEILING at 1.5x nproc (24 on qb2). Fanning the gate
#    out over three cards pushed loadavg to 30.85 and every arm after the first refused to run.
#    So wait for the box to fall below CEIL before each arm instead of assuming an accuracy
#    gate does not care about load. It cares: not about the verdict, about whether it starts.
#  - the resume marker was written regardless of exit status, so a refused arm counted as done
#    and a re-run skipped it. Only a rc=0 PASS marks an arm done; anything else is retried.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/land_standing/out/gate_d10d24
CARD=$1; CEIL=$2; shift 2
DEADLINE=$(( $(date +%s) + 9000 ))
mkdir -p "$OUT"
cd "$WT" || exit 1

load1() { awk '{print int($1)}' /proc/loadavg; }

for round in 1 2 3 4; do
  left=0
  for arm in "$@"; do
    [ -f "$OUT/done_$arm" ] && continue
    while [ "$(load1)" -ge "$CEIL" ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 30; done
    t0=$(date +%s)
    TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing \
      PYTHONPATH=$WT RELEASE_GATE_CENSUS_PYTHONPATH=$WT \
      "$PY" scripts/release_gate.py --model "$arm" > "$OUT/$arm.log" 2>&1
    rc=$?
    t1=$(date +%s)
    v=$(grep -Eo "\b(PASS|FAIL|SKIP)\b" "$OUT/$arm.log" | tail -1)
    echo "$arm rc=$rc verdict=${v:-none} wall=$((t1-t0))s card=$CARD round=$round $(date -u +%FT%TZ)" >> "$OUT/SUMMARY.txt"
    if [ "$rc" -eq 0 ] && [ "$v" = "PASS" ]; then touch "$OUT/done_$arm"; else left=1; sleep 60; fi
  done
  [ "$left" -eq 0 ] && break
done
echo "SHARD card=$CARD DONE $(date -u +%FT%TZ)" >> "$OUT/SUMMARY.txt"

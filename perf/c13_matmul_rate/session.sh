#!/usr/bin/env bash
# One clean timed session for c13-matmul-rate on qb2 node 3 (board ...410D; node 2 is held idle as
# its board-pair sibling by state/cardblock-qb2-2, so this is a valid timing pair).
#
# The clock is FORCED by rate.py itself, inside the measuring process, and sampled by a process
# that holds no device fd. Session s1 forced and sampled from one helper, and its record could not
# distinguish a clean 21 s session from a clean 21 s record of a much longer measurement, so
# reduce.py now refuses any session whose sampler span does not cover the measurement span.
#
# Node 3 is verified by fuser, never by the UMD device id: UMD calls this chip device 1 while it
# is /dev/tenstorrent/3.
set -u
WT=/home/ttuser/.coworker/wt/c13-matmul-rate
OUT=$WT/perf/c13_matmul_rate
PY=/home/ttuser/tt-bio-dev/env/bin/python3
TAG="${1:-s2}"; shift || true
cd "$WT"

$PY "$OUT/sample_aiclk.py" --nodes 3 --out "$OUT/clock_$TAG.jsonl" > "$OUT/clock_$TAG.log" 2>&1 &
SAMP=$!
(
  while kill -0 $SAMP 2>/dev/null; do
    n0=$(sudo fuser /dev/tenstorrent/0 2>&1 | tr -d '\n')
    n2=$(sudo fuser /dev/tenstorrent/2 2>&1 | tr -d '\n')
    n3=$(sudo fuser /dev/tenstorrent/3 2>&1 | tr -d '\n')
    echo "$(date +%s) n0=[$n0] n2=[$n2] n3=[$n3]"
    sleep 2
  done > "$OUT/fuser_$TAG.log" 2>&1
) &
WATCH=$!
sleep 2

/home/ttuser/.coworker/scripts/benchlock.sh c13-matmul-rate -- \
  env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:c13-matmul-rate \
  "$PY" "$OUT/rate.py" --clock 1350 --node 3 --out "$OUT/rate_$TAG.json" "$@"
RC=$?
echo "rate.py rc=$RC"
kill -TERM $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
kill -TERM $WATCH 2>/dev/null

echo "=== distinct device occupancy seen over the session ==="
awk '{ $1=""; print }' "$OUT/fuser_$TAG.log" | sort -u | head -20
if [ "$RC" = "0" ]; then
  "$PY" "$OUT/reduce.py" --rate "$OUT/rate_$TAG.json" --clock "$OUT/clock_$TAG.jsonl" \
        --out "$OUT/decomp_$TAG.json"
fi
exit $RC

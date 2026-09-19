#!/usr/bin/env bash
# One clean timed session for c14-matmul-ceiling on qb2 node 3 (board ...410D; node 2 is held idle
# as its board-pair sibling by state/cardblock-qb2-2, so this is a valid timing pair). Node 0 is
# occupied by another worker's job and node 1 is dead, so node 3 is the only timing chip free.
#
# The clock is FORCED inside the measuring process and sampled by a process holding no device fd.
# Node identity is checked with fuser, never with the UMD device id: UMD calls this chip device 1.
set -u
WT=/home/ttuser/.coworker/wt/c14-matmul-ceiling
OUT=$WT/perf/c14_matmul_ceiling
C13=/tmp/c14mc/perf/c13_matmul_rate
PY=/home/ttuser/tt-bio-dev/env/bin/python3
TAG="${1:-s1}"; shift || true
cd "$WT"

$PY "$C13/sample_aiclk.py" --nodes 3 --out "$OUT/clock_$TAG.jsonl" > "$OUT/clock_$TAG.log" 2>&1 &
SAMP=$!
(
  while kill -0 $SAMP 2>/dev/null; do
    n0=$(fuser /dev/tenstorrent/0 2>&1 | tr -d '\n')
    n2=$(fuser /dev/tenstorrent/2 2>&1 | tr -d '\n')
    n3=$(fuser /dev/tenstorrent/3 2>&1 | tr -d '\n')
    echo "$(date +%s) n0=[$n0] n2=[$n2] n3=[$n3]"
    sleep 2
  done > "$OUT/fuser_$TAG.log" 2>&1
) &
WATCH=$!
sleep 2

/home/ttuser/.coworker/scripts/benchlock.sh c14-matmul-ceiling -- \
  env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:c14-matmul-ceiling \
  "$PY" "$OUT/ditrate.py" --clock 1350 --node 3 --tag "$TAG" "$@"
RC=$?
echo "ditrate.py rc=$RC"
kill -TERM $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
kill -TERM $WATCH 2>/dev/null
echo "=== distinct device occupancy over the session ==="
awk '{ $1=""; print }' "$OUT/fuser_$TAG.log" | sort -u | head -20
exit $RC

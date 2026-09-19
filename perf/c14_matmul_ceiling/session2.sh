#!/usr/bin/env bash
set -u
WT=/home/ttuser/.coworker/wt/c14-matmul-ceiling
OUT=$WT/perf/c14_matmul_ceiling
C13=/tmp/c14mc/perf/c13_matmul_rate
PY=/home/ttuser/tt-bio-dev/env/bin/python3
TAG="${1:-b1}"; shift || true
NODE="${NODE:-3}"
cd "$WT"
python3 perf/c12_orchestrator/pair_guard/pair_idle.py --card "$NODE" || { echo "PAIR GUARD REFUSED"; exit 75; }
$PY "$C13/sample_aiclk.py" --nodes "$NODE" --out "$OUT/clock_$TAG.jsonl" > "$OUT/clock_$TAG.log" 2>&1 &
SAMP=$!
( while kill -0 $SAMP 2>/dev/null; do
    echo "$(date +%s) n0=[$(fuser /dev/tenstorrent/0 2>&1|tr -d '\n')] n2=[$(fuser /dev/tenstorrent/2 2>&1|tr -d '\n')] n3=[$(fuser /dev/tenstorrent/3 2>&1|tr -d '\n')]"
    sleep 2
  done > "$OUT/fuser_$TAG.log" 2>&1 ) &
WATCH=$!
sleep 2
/home/ttuser/.coworker/scripts/benchlock.sh c14-matmul-ceiling -- \
  env TT_VISIBLE_DEVICES=$NODE TT_BIO_LEASE_CARDS=$NODE TT_BIO_LEASE_HOLDER=worker:c14-matmul-ceiling \
  "$PY" "$OUT/bwladder.py" --clock 1350 --node "$NODE" --tag "$TAG" "$@"
RC=$?
echo "bwladder.py rc=$RC"
kill -TERM $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
kill -TERM $WATCH 2>/dev/null
echo "=== distinct device occupancy over the session ==="
awk '{ $1=""; print }' "$OUT/fuser_$TAG.log" | sort -u | head -20
exit $RC

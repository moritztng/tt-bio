#!/bin/bash
# Localize the 1024 aa ON-arm spin: run one leg, sample the Python stack while it spins.
set -u
WT=/home/ttuser/.coworker/wt/roof-transition-l1-1024-overflow-fix
OUT=$WT/perf/roof_transition_l1_1024/out
PY=/home/ttuser/tt-bio-dev/env/bin/python3
STEPS=${STEPS:-2}
CAP=${CAP:-420}
TAG=${TAG:-spin}
LEGS=${LEGS:-1024:on}
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 \
       TT_BIO_LEASE_HOLDER=worker:roof-transition-l1-1024-overflow-fix \
       TT_BIO_TRANSITION_TRACE=1
timeout -s KILL "$CAP" "$PY" perf/roof_transition_chunk_bh/foldab.py \
    --out "$OUT/${TAG}.json" --legs "$LEGS" --reps 1 --steps "$STEPS" \
    > "$OUT/${TAG}.log" 2>&1 &
RUN=$!
echo "launched pid=$RUN legs=$LEGS steps=$STEPS cap=$CAP"
for t in 60 150 260 380; do
  while [ "$SECONDS" -lt "$t" ]; do
    sleep 5
    kill -0 "$RUN" 2>/dev/null || { echo "=== run exited at t=${SECONDS}s ==="; break 2; }
  done
  KID=$(pgrep -P "$RUN" -f foldab.py | head -1)
  TGT=${KID:-$RUN}
  echo "===== py-spy t=${t}s pid=$TGT cpu=$(ps -o pcpu= -p "$TGT" 2>/dev/null | tr -d " ") ====="
  /home/ttuser/.local/bin/py-spy dump --pid "$TGT" --nonblocking 2>&1 | head -55
done
wait "$RUN"; rc=$?
echo "=== rc=$rc elapsed=${SECONDS}s ==="
echo "===== tail of log ====="
tail -35 "$OUT/${TAG}.log"

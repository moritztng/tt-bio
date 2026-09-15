#!/bin/bash
# Reproduce the 1024 aa wedge and capture what the process is actually doing when it stops.
# Wedge = the transition trace stops advancing while the process stays alive and busy.
set -u
WT=/home/ttuser/.coworker/wt/roof-transition-l1-1024-overflow-fix
OUT=$WT/perf/roof_transition_l1_1024/out
PY=/home/ttuser/tt-bio-dev/env/bin/python3
TAG=${TAG:-wedge}
STEPS=${STEPS:-2}
CAP=${CAP:-1500}
STALL=${STALL:-90}
LEGS=${LEGS:-1024:on}
LOG=$OUT/${TAG}.log
CAP_OUT=$OUT/${TAG}.capture.txt
cd "$WT" || exit 1
: > "$CAP_OUT"
env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 \
    TT_BIO_LEASE_HOLDER=worker:roof-transition-l1-1024-overflow-fix \
    TT_BIO_TRANSITION_TRACE=1 \
    timeout -s KILL "$CAP" "$PY" perf/roof_transition_chunk_bh/foldab.py \
      --out "$OUT/${TAG}.json" --legs "$LEGS" --reps 1 --steps "$STEPS" \
      > "$LOG" 2>&1 &
RUN=$!
echo "launched pid=$RUN legs=$LEGS steps=$STEPS cap=$CAP stall=$STALL" | tee -a "$CAP_OUT"
last_n=-1; still=0
while kill -0 "$RUN" 2>/dev/null; do
  sleep 15
  n=$(grep -c "\[transition-h\]" "$LOG" 2>/dev/null || echo 0)
  if [ "$n" -eq "$last_n" ] && [ "$n" -gt 0 ]; then
    still=$((still+15))
  else
    still=0
  fi
  last_n=$n
  echo "t=${SECONDS}s trace_lines=$n stalled_for=${still}s load=$(cut -d" " -f1 /proc/loadavg)" | tee -a "$CAP_OUT"
  if [ "$still" -ge "$STALL" ]; then
    KID=$(pgrep -P "$RUN" -f foldab.py | head -1); TGT=${KID:-$RUN}
    {
      echo "================ WEDGE CAPTURE t=${SECONDS}s pid=$TGT ================"
      echo "--- last trace line ---"; grep "\[transition-h\]" "$LOG" | tail -2
      echo "--- proc ---"; ps -o pid,ppid,pcpu,stat,etimes,nlwp -p "$TGT"
      echo "--- per-thread cpu ---"; ps -L -o tid,pcpu,stat,wchan:24 -p "$TGT" | head -25
      echo "--- py-spy ---"; /home/ttuser/.local/bin/py-spy dump --pid "$TGT" --nonblocking 2>&1 | head -30
      echo "--- gdb native bt (busiest threads) ---"
      timeout 180 gdb -p "$TGT" -batch -ex "set pagination off" -ex "thread apply all bt 22" 2>&1 | head -220
      echo "--- tt-smi ---"; timeout 60 ~/.local/bin/tt-smi -s 2>&1 | grep -iE "chip|arc|status|temp|power" | head -20
    } >> "$CAP_OUT" 2>&1
    echo "captured; killing $RUN"; kill -9 "$RUN" 2>/dev/null
    pkill -9 -f "foldab.py --out $OUT/${TAG}.json" 2>/dev/null
    break
  fi
done
echo "=== done elapsed=${SECONDS}s ===" | tee -a "$CAP_OUT"

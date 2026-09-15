#!/bin/bash
# v2: name the wedge. Two instruments the first capture lacked.
#  - TTNN_CONFIG_OVERRIDES throw_exception_on_fallback: if a ttnn op is silently running on the
#    host, this turns it into a Python traceback that names the op instead of a 100% CPU spin.
#  - gdb on THREAD 1 ONLY. v1 asked for every thread and the 41 idle ones pushed the main
#    thread's frames past the output cap, which is the one stack that mattered.
set -u
WT=/home/ttuser/.coworker/wt/roof-transition-l1-1024-overflow-fix
OUT=$WT/perf/roof_transition_l1_1024/out
PY=/home/ttuser/tt-bio-dev/env/bin/python3
TAG=${TAG:-wedge2}
STEPS=${STEPS:-2}
CAP=${CAP:-1200}
STALL=${STALL:-75}
LEGS=${LEGS:-1024:on}
REF=${REF:-ship}
FALLBACK=${FALLBACK:-0}
LOG=$OUT/${TAG}.log
CAP_OUT=$OUT/${TAG}.capture.txt
cd "$WT" || exit 1
: > "$CAP_OUT"
EXTRA=()
if [ "$FALLBACK" = "1" ]; then
  EXTRA=(TTNN_CONFIG_OVERRIDES={\"throw_exception_on_fallback\":true})
fi
env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 \
    TT_BIO_LEASE_HOLDER=worker:roof-transition-l1-1024-overflow-fix \
    TT_BIO_TRANSITION_TRACE=1 "${EXTRA[@]}" \
    timeout -s KILL "$CAP" "$PY" perf/roof_transition_chunk_bh/foldab.py \
      --out "$OUT/${TAG}.json" --legs "$LEGS" --ref "$REF" --reps 1 --steps "$STEPS" \
      > "$LOG" 2>&1 &
RUN=$!
echo "launched pid=$RUN legs=$LEGS steps=$STEPS fallback_throw=$FALLBACK" | tee -a "$CAP_OUT"
last_n=-1; still=0
while kill -0 "$RUN" 2>/dev/null; do
  sleep 15
  n=$(grep -c "transition-h" "$LOG" 2>/dev/null || echo 0)
  if [ "$n" -eq "$last_n" ] && [ "$n" -gt 0 ]; then still=$((still+15)); else still=0; fi
  last_n=$n
  echo "t=${SECONDS}s trace=$n stalled=${still}s load=$(cut -d' ' -f1 /proc/loadavg)" | tee -a "$CAP_OUT"
  if [ "$still" -ge "$STALL" ]; then
    KID=$(pgrep -P "$RUN" -f foldab.py | head -1); TGT=${KID:-$RUN}
    {
      echo "======== WEDGE t=${SECONDS}s pid=$TGT ========"
      echo "--- py-spy ---"; /home/ttuser/.local/bin/py-spy dump --pid "$TGT" --nonblocking 2>&1 | head -26
      echo "--- gdb THREAD 1 ---"
      timeout 200 gdb -p "$TGT" -batch -ex "set pagination off" -ex "thread 1" -ex "bt 40" 2>&1 \
        | grep -vE "^\[New LWP|^warning:" | head -70
      echo "--- gdb THREAD 1 again (is it advancing?) ---"
      timeout 200 gdb -p "$TGT" -batch -ex "set pagination off" -ex "thread 1" -ex "bt 12" 2>&1 \
        | grep -vE "^\[New LWP|^warning:" | head -25
    } >> "$CAP_OUT" 2>&1
    kill -9 "$RUN" 2>/dev/null
    pkill -9 -f "foldab.py --out $OUT/${TAG}.json" 2>/dev/null
    break
  fi
done
echo "=== done elapsed=${SECONDS}s log_tail ===" | tee -a "$CAP_OUT"
grep -v "Initial ttnn.CONFIG" "$LOG" | grep -v "^Config{" | tail -25 | tee -a "$CAP_OUT"

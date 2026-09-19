#!/bin/sh
# One OpenDDE coverage rung on one Blackhole p150a, with the clock it was measured at.
#
# On Blackhole the AICLK sets the fold time, so a wall-clock without a clock beside it is not a
# measurement. The sampler reads every card's sysfs AICLK every 10 s DURING the fold: an idle card
# decays to 800 and ramps when work lands on it, so a reading taken at launch describes nothing.
# All four nodes are sampled rather than just ours, because the tt-smi UMD id in TT_VISIBLE_DEVICES
# is not the /dev/tenstorrent node number (UMD 0 is node1 on this host).
#
# LIVENESS IS NOT LOG GROWTH ON THIS MODEL. The 2026-09-10 harness (perf/bh1536/run_rung.py) judged
# a rung STALLED when its log stopped growing for 600 s, which is how OpenDDE's Blackhole freeze was
# caught. On the same flags a HEALTHY 1536 fold here writes nothing for 8 minutes at a stretch:
# --debug at TT_METAL_LOGGER_LEVEL=FATAL leaves whole phases silent, and the first fold of a shape
# spends minutes compiling kernels. So the watch below samples the WORKER's Python frame with py-spy
# and calls a rung frozen only when the frame, the RSS and the log are all unchanged across the
# whole window -- the three-way signature the freeze actually had (100 % CPU on one core, RSS pinned
# to the byte, no allocator refusal anywhere). py-spy needs `sudo -n` on this host; without it every
# dump is one Permission Denied line and the watch is blind.
#
#   sh rung.sh <name> <umd_device> <mode: msa|single> [budget_s] [stall_s]
set -u
WT=/home/ttuser/.coworker/wt/cov-unproven-opendde-bhp150a
PY=/home/ttuser/tt-bio/env/bin/python3
SPY="sudo -n env PATH=$PATH /home/ttuser/.local/bin/py-spy"
B=$WT/perf/oddcov
NAME=$1; DEV=$2; MODE=$3; BUDGET=${4:-3000}; STALL=${5:-900}
CKPT=${CKPT:-opendde}
OUT=$B/out/${CKPT}_${NAME}_${MODE}_dev${DEV}
rm -rf "$OUT"; mkdir -p "$OUT"
LOG=$OUT/fold.log; CLK=$OUT/aiclk.log; RAM=$OUT/host.log

( while :; do
    line=$(date +%s)
    for n in 0 1 2 3; do
      line="$line $(cat /sys/class/tenstorrent/tenstorrent\!$n/tt_aiclk 2>/dev/null || echo NA)"
    done
    echo "$line" >> "$CLK"
    printf '%s %s %s\n' "$(date +%s)" "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" \
      "$(cut -d' ' -f1 /proc/loadavg)" >> "$RAM"
    sleep 10
  done ) &
SIDE=$!

if [ "$MODE" = single ]; then MSAARG="--single_sequence"; else MSAARG="--msa_dir $B/msa --msa_cache_only"; fi

s=$(date +%s)
# TT_BIO_SIZE_LIMIT=0 turns the engine's own blackhole opendde refusal (1024 residues, set from
# the very freeze this rung re-measures) into a warning. Without it the 1536 rung is declined by
# the guard and nothing is measured.
setsid env TT_VISIBLE_DEVICES=$DEV TT_BIO_LEASE_CARDS=$DEV \
    TT_BIO_LEASE_HOLDER=worker:cov-unproven-opendde-bhp150a \
    TT_BIO_SIZE_LIMIT=0 TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH="$WT" \
    "$PY" -m tt_bio.main predict "$B/inputs/odd_${NAME}.yaml" \
      --model "$CKPT" --accelerator tenstorrent $MSAARG \
      --out_dir "$OUT" --override --debug > "$LOG" 2>&1 &
PID=$!
cleanup() { kill "$SIDE" 2>/dev/null; kill -9 -"$PID" 2>/dev/null; }
trap cleanup EXIT INT TERM

# The fold runs in a spawned worker, so the frame that matters belongs to a child of $PID.
worker() { pgrep -P "$PID" -f "tt_bio" | head -1; }

waited=0; quiet=0; lastlog=0; lastframe=; lastrss=; verdict=RAN
while kill -0 "$PID" 2>/dev/null; do
  sleep 30; waited=$((waited + 30))
  w=$(worker)
  now=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  frame=$($SPY dump --pid "${w:-$PID}" 2>/dev/null | sed -n '4,8p')
  rss=$(awk '{print $2}' /proc/"${w:-$PID}"/statm 2>/dev/null)
  if [ "$now" = "$lastlog" ] && [ "$frame" = "$lastframe" ] && [ "$rss" = "$lastrss" ] \
     && [ -n "$frame" ]; then
    quiet=$((quiet + 30))
  else
    quiet=0
  fi
  lastlog=$now; lastframe=$frame; lastrss=$rss
  if [ "$quiet" -ge "$STALL" ]; then
    verdict=FROZEN
    tail -5 "$LOG" > "$OUT/last_progress.txt"
    for i in 1 2 3; do
      { $SPY dump --pid "${w:-$PID}"; ps -L -o pid,tid,stat,wchan:24,pcpu -p "${w:-$PID}"; } \
        > "$OUT/pyspy-$i.txt" 2>&1
      sleep 20
    done
    kill -9 -"$PID" 2>/dev/null
    break
  fi
  if [ "$waited" -ge "$BUDGET" ]; then
    verdict=TIMEOUT
    tail -5 "$LOG" > "$OUT/last_progress.txt"
    $SPY dump --pid "${w:-$PID}" > "$OUT/pyspy-timeout.txt" 2>&1
    kill -9 -"$PID" 2>/dev/null
    break
  fi
done
wait "$PID" 2>/dev/null; rc=$?
e=$(date +%s)
kill "$SIDE" 2>/dev/null
echo "RUNG name=$NAME ckpt=$CKPT mode=$MODE dev=$DEV rc=$rc verdict=$verdict wall_s=$((e - s))" \
  | tee "$OUT/rung.txt"

#!/bin/sh
# One rung of the Blackhole 1024 ladder: fold <rung>.yaml with <model> at the platform's
# resolved recycling/sampling, on pc card 0, and record capacity + wall-clock + both memory axes.
#
# Two arms, and they must not be conflated:
#   probe=off  -> the wall-clock arm. TT_BIO_DRAM_PEAK unset, so the fold is not slowed.
#   probe=on   -> the footprint arm. TT_BIO_DRAM_PEAK writes peak device DRAM, but the
#                 allocator probe drains the pipeline and costs ~2.4x wall-clock
#                 (tenstorrent.py:dram_peak says so outright), so this arm's wall is NOT a
#                 timing measurement and is recorded under a different key.
#
# The fold runs in its OWN process group (setsid) and is killed by group, never by the outer
# pid. A plain `timeout`, or worker.sh's pass-end SIGTERM, reaches only the launcher: pass 1
# lost a 52-minute rung exactly that way, and the multiprocessing engine child kept the card
# for another hour after its parent was gone with nothing left to record the result.
#
#   sh run_rung.sh <model> <rung> <recycling> <sampling> <off|on>
set -u
WT=/home/moritz/.coworker/wt/bh-1024-of3-openbind-rf3
PY=/home/moritz/tt-bio/env/bin/python3
B=$WT/perf/bh1024
MODEL=$1; RUNG=$2; RECYC=$3; SAMP=$4; PROBE=${5:-off}
# A rung that has not finished in this long is not going to: Wormhole folds 768 at the same
# 14190-row depth in 365 s, so 4500 s is a >12x budget. Bounded on purpose -- an unbounded
# rung is what let one wedge eat a whole pass.
BUDGET=${RUNG_TIMEOUT:-4500}
TAG="${MODEL}_${RUNG}_${PROBE}"
OUT=$B/out/$TAG
mkdir -p "$OUT"
LOG=$OUT/fold.log
RAMLOG=$OUT/hostram.log
PEAK=$OUT/dram_peak.log
STACK=$OUT/stack.log
rm -f "$LOG" "$RAMLOG" "$PEAK" "$STACK"

# Host RAM sidecar. pc has 30 GB against the Galaxy's 566, and a host OOM here is a
# DIFFERENT result from the device DRAM wall this ladder is about, so both axes get sampled
# rather than inferred from whichever one failed.
( while :; do
    printf '%s %s\n' "$(date +%s)" "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" >> "$RAMLOG"
    ps -o rss= -C python3 2>/dev/null | awk '{s+=$1} END{printf "RSS %d\n", s}' >> "$RAMLOG"
    sleep 2
  done ) &
RAMPID=$!

if [ "$PROBE" = on ]; then PEAKENV="TT_BIO_DRAM_PEAK=$PEAK"; else PEAKENV="TT_BIO_DRAM_PEAK="; fi

s=$(date +%s)
# setsid => the fold and every engine child it spawns share one process group we can kill.
setsid env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
    TT_BIO_LEASE_HOLDER=worker:bh-1024-of3-openbind-rf3 \
    TT_METAL_LOGGER_LEVEL=FATAL "$PEAKENV" \
    PYTHONPATH="$WT" \
    "$PY" -m tt_bio.main predict "$B/inputs/$RUNG.yaml" \
      --model "$MODEL" --accelerator tenstorrent \
      --recycling_steps "$RECYC" --sampling_steps "$SAMP" \
      --msa_dir "$B/msa_deep" --msa_cache_only \
      --out_dir "$OUT" --override --debug > "$LOG" 2>&1 &
FOLDPID=$!
PGID=$FOLDPID   # setsid makes the child a group leader, so pgid == pid

cleanup() { kill "$RAMPID" 2>/dev/null; kill -9 -"$PGID" 2>/dev/null; }
trap 'cleanup' EXIT INT TERM

# Watchdog. Samples the deepest python stack every 5 min so a stall is attributable to an op
# rather than just "it did not finish" -- pass 1 could only tell a wedge from slow compute by
# reading a live stack, so the ladder now keeps that receipt for every rung.
killed_by_budget=0
waited=0
while kill -0 "$FOLDPID" 2>/dev/null; do
  sleep 20
  waited=$((waited + 20))
  if [ $((waited % 300)) -eq 0 ]; then
    { echo "--- t=${waited}s $(date -u +%FT%TZ) ---"
      deep=$(pgrep -g "$PGID" -f 'multiprocessing.spawn|tt_bio' 2>/dev/null | tail -1)
      [ -n "$deep" ] && timeout 60 /home/moritz/.local/bin/py-spy dump --pid "$deep" --locals 2>/dev/null \
        | grep -E '^Thread|^ +[a-z_]+ \(|^ +tau: |^ +n_token: ' | head -25
    } >> "$STACK" 2>&1
  fi
  if [ "$waited" -ge "$BUDGET" ]; then
    { echo "=== BUDGET ${BUDGET}s EXCEEDED, final stack $(date -u +%FT%TZ) ==="
      deep=$(pgrep -g "$PGID" -f 'multiprocessing.spawn|tt_bio' 2>/dev/null | tail -1)
      [ -n "$deep" ] && timeout 60 /home/moritz/.local/bin/py-spy dump --pid "$deep" --locals 2>/dev/null | head -40
    } >> "$STACK" 2>&1
    killed_by_budget=1
    kill -9 -"$PGID" 2>/dev/null
    break
  fi
done
wait "$FOLDPID" 2>/dev/null
rc=$?
[ "$killed_by_budget" -eq 1 ] && rc=124
e=$(date +%s)
kill "$RAMPID" 2>/dev/null
trap - EXIT INT TERM
kill -9 -"$PGID" 2>/dev/null   # sweep any engine child that outlived the launcher

"$PY" "$B/record.py" "$MODEL" "$RUNG" "$RECYC" "$SAMP" "$PROBE" "$rc" "$((e - s))" "$OUT"

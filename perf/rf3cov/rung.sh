#!/bin/sh
# One RoseTTAFold3 coverage rung on one Blackhole p150a, with the clock it was measured at.
#
# On Blackhole the AICLK sets the fold time, so a wall clock without a clock beside it is not a
# measurement. The sampler reads every card's sysfs AICLK every 10 s DURING the fold: an idle
# card decays to 800 MHz and ramps when work lands, so a reading taken at launch describes
# nothing. All four nodes are sampled because the tt-smi UMD id is not the /dev/tenstorrent node
# number on this host (UMD 0 is node1).
#
# RSS of the whole process group is sampled alongside, because a stall is not a pass. The prior
# rf3 run at 1536 tokens went from "trunk 3/10" to gone in 30 s after eight minutes of folding
# and returned 0 with an empty structures/ (CHANGELOG, "A fold stopped by a signal now says so"),
# so a rung needs a forward-progress signal and an exit code, not an exit code alone. The group
# sum is what gets sampled and not the launched pid: that pid is a click front end sitting at a
# few hundred MB while a spawned worker does all the work at tens of GB.
#
# Flags are the shipped defaults on purpose: no --recycling_steps, no --sampling_steps, no
# --max_msa_seqs. rf3 resolves 10 recycles and 50 sampling steps from its own tables and folds
# the resolved alignment whole, so passing any of the three would measure a configuration no
# user is in. --msa_cache_only makes the staged msa/ the only MSA source, so no rung can quietly
# fall back to a search and fold a different depth than the one recorded.
#
#   sh rung.sh <rung> <umd_device> [budget_s]
set -u
WT=/home/ttuser/.coworker/wt/cov-below-bar-rf3-bhp150a
PY=/home/ttuser/tt-bio/env/bin/python3
B=$WT/perf/rf3cov
RUNG=$1; DEV=$2; BUDGET=${3:-3000}
OUT=$B/out/rf3_${RUNG}_dev${DEV}
rm -rf "$OUT"; mkdir -p "$OUT"
LOG=$OUT/fold.log; CLK=$OUT/aiclk.log; RAM=$OUT/host.log

s=$(date +%s)
setsid env TT_VISIBLE_DEVICES=$DEV TT_BIO_LEASE_CARDS=$DEV \
    TT_BIO_LEASE_HOLDER=worker:cov-below-bar-rf3-bhp150a \
    TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH="$WT" \
    "$PY" -m tt_bio.main predict "$B/inputs/rf3_${RUNG}.yaml" \
      --model rf3 --accelerator tenstorrent \
      --msa_dir "$B/msa" --msa_cache_only \
      --out_dir "$OUT" --override --debug > "$LOG" 2>&1 &
PID=$!

( while :; do
    line=$(date +%s)
    for n in 0 1 2 3; do
      line="$line $(cat /sys/class/tenstorrent/tenstorrent\!$n/tt_aiclk 2>/dev/null || echo NA)"
    done
    echo "$line" >> "$CLK"
    grp=$(ps -o rss= -g "$PID" 2>/dev/null | awk '{t+=$1} END {print t+0}')
    printf '%s %s %s %s\n' "$(date +%s)" "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" \
      "$(cut -d' ' -f1 /proc/loadavg)" "$grp" >> "$RAM"
    sleep 10
  done ) &
SIDE=$!

cleanup() { kill "$SIDE" 2>/dev/null; kill -9 -"$PID" 2>/dev/null; }
trap cleanup EXIT INT TERM

waited=0
while kill -0 "$PID" 2>/dev/null; do
  sleep 20; waited=$((waited + 20))
  [ "$waited" -ge "$BUDGET" ] && { echo "BUDGET $BUDGET s reached" >> "$LOG"; kill -9 -"$PID"; break; }
done
wait "$PID" 2>/dev/null; rc=$?
e=$(date +%s)
kill "$SIDE" 2>/dev/null
echo "RUNG rung=$RUNG dev=$DEV rc=$rc wall_s=$((e - s))" >> "$OUT/rung.txt"
echo "RUNG rung=$RUNG dev=$DEV rc=$rc wall_s=$((e - s))"

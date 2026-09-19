#!/bin/sh
# One OpenFold3 coverage rung on one Blackhole p150a, with the clock it was measured at.
#
# On Blackhole the AICLK sets the fold time, so a wall clock without a clock beside it is not a
# measurement. The sampler reads every card's sysfs AICLK every 10 s DURING the fold: an idle
# card decays to 800 MHz and ramps when work lands, so a reading taken at launch describes
# nothing. All four nodes are sampled because the tt-smi UMD id is not the /dev/tenstorrent node
# number on this host (UMD 0 is node1).
#
# OpenFold3 can stall rather than throw. The L1 refusal inside its diffusion transformer is
# caught and retried, which once burned 2289 s at diffusion step 0, so a rung that runs to its
# budget needs a forward-progress signal and not just an exit status. Flat group RSS with the
# log unchanged is that signal, and it is sampled by grouprss.sh rather than from the pid this
# script launches, which is a click front end that sits flat for the whole fold.
#
# Flags are the shipped defaults on purpose: no --recycling_steps, no --sampling_steps, no
# --max_msa_seqs, because the OF3 family folds the resolved alignment whole and applying the
# boltz2 8192 default would measure a configuration no user is in. --msa_cache_only makes the
# staged msa/ the only MSA source, so no rung can quietly fall back to a search and fold a
# different depth than the one recorded.
#
#   sh rung.sh <rung> <umd_device> [budget_s]
set -u
WT=/home/ttuser/.coworker/wt/cov-below-bar-openfold3-bhp150a
PY=/home/ttuser/tt-bio/env/bin/python3
B=$WT/perf/of3cov
RUNG=$1; DEV=$2; BUDGET=${3:-5400}
OUT=$B/out/of3_${RUNG}_dev${DEV}
rm -rf "$OUT"; mkdir -p "$OUT"
LOG=$OUT/fold.log; CLK=$OUT/aiclk.log; RAM=$OUT/host.log

s=$(date +%s)
setsid env TT_VISIBLE_DEVICES=$DEV TT_BIO_LEASE_CARDS=$DEV \
    TT_BIO_LEASE_HOLDER=worker:cov-below-bar-openfold3-bhp150a \
    TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH="$WT" \
    "$PY" -m tt_bio.main predict "$B/inputs/of3_${RUNG}.yaml" \
      --model openfold3 --accelerator tenstorrent \
      --msa_dir "$B/msa" --msa_cache_only \
      --out_dir "$OUT" --override --debug > "$LOG" 2>&1 &
PID=$!

( while :; do
    line=$(date +%s)
    for n in 0 1 2 3; do
      line="$line $(cat /sys/class/tenstorrent/tenstorrent\!$n/tt_aiclk 2>/dev/null || echo NA)"
    done
    echo "$line" >> "$CLK"
    printf '%s %s %s %s\n' "$(date +%s)" "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" \
      "$(cut -d' ' -f1 /proc/loadavg)" \
      "$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ' || echo NA)" >> "$RAM"
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

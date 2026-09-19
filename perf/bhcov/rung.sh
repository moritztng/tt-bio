#!/bin/sh
# One boltz2 coverage rung on one Blackhole p150a, with the clock it was measured at.
#
# On Blackhole the AICLK sets the fold time -- 512 aa reads 21.90 s at 800 MHz and 14.69 s at
# 1350 -- so a wall-clock without a clock beside it is not a measurement. The sampler reads
# every card's sysfs AICLK every 10 s DURING the fold, not before: an idle card decays to 800
# and ramps when work lands on it, so a reading taken at launch describes nothing.
#
# All four nodes are sampled rather than just ours, because the tt-smi UMD id in
# TT_VISIBLE_DEVICES is not the /dev/tenstorrent node number (UMD 0 is node1 on this host).
# Sampling all four and reporting the one under load costs nothing and cannot be mis-attributed.
#
#   sh rung.sh <tokens> <umd_device> [budget_s]
set -u
WT=/home/ttuser/.coworker/wt/cov-unproven-boltz2-bhp150a
PY=/home/ttuser/tt-bio/env/bin/python3
B=$WT/perf/bhcov
TOK=$1; DEV=$2; BUDGET=${3:-4200}
OUT=$B/out/b2_${TOK}_dev${DEV}
rm -rf "$OUT"; mkdir -p "$OUT"
LOG=$OUT/fold.log; CLK=$OUT/aiclk.log; RAM=$OUT/host.log

( while :; do
    line=$(date +%s)
    for n in 0 1 2 3; do
      line="$line $(cat /sys/class/tenstorrent/tenstorrent\!$n/tt_aiclk 2>/dev/null || echo NA)"
    done
    echo "$line" >> "$CLK"
    printf '%s %s %s\n' "$(date +%s)" "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" \
      "$(cat /proc/loadavg | cut -d' ' -f1)" >> "$RAM"
    sleep 10
  done ) &
SIDE=$!

s=$(date +%s)
setsid env TT_VISIBLE_DEVICES=$DEV TT_BIO_LEASE_CARDS=$DEV \
    TT_BIO_LEASE_HOLDER=worker:cov-unproven-boltz2-bhp150a \
    TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH="$WT" \
    "$PY" -m tt_bio.main predict "$B/inputs/b2_${TOK}.yaml" \
      --model boltz2 --accelerator tenstorrent \
      --msa_dir "$B/msa" --msa_cache_only \
      --out_dir "$OUT" --override --debug > "$LOG" 2>&1 &
PID=$!
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
echo "RUNG tokens=$TOK dev=$DEV rc=$rc wall_s=$((e - s))" >> "$OUT/rung.txt"
echo "RUNG tokens=$TOK dev=$DEV rc=$rc wall_s=$((e - s))"

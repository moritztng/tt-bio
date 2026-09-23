#!/usr/bin/env bash
# One device_gradient arm, with the clock sampled DURING it.
#
# On Blackhole the AICLK sets the runtime, and an idle card decays to 800 MHz, so a wall clock
# taken beside a clock read BEFORE the run is not a measurement. The sampler runs for the arm's
# whole life and the arm's record carries min/median/max over its own window.
#
# `--ref-tree` and the capture's REFTREE.json stamp are what make this arm's DENOMINATOR a
# reading. of3t-ditcot's three arms have neither.
set -uo pipefail
WT=/home/ttuser/.coworker/wt/of3t-ditref
cd "$WT"
SCR=/tmp/of3t/of3t-ditref
mkdir -p "$SCR"
PY=/home/ttuser/tt-bio-dev/env/bin/python
DEV=1

TAG=$1; shift
CLK=$SCR/aiclk_$TAG.log
LOG=$SCR/$TAG.log
: > "$CLK"

( while :; do
    echo "$(date +%s) $(cat /sys/class/tenstorrent/tenstorrent\!$DEV/tt_aiclk 2>/dev/null || echo NA)" >> "$CLK"
    sleep 5
  done ) &
SIDE=$!
trap 'kill $SIDE 2>/dev/null' EXIT

L0=$(cut -d' ' -f1-3 /proc/loadavg)
S=$(date +%s)
echo "=== arm $TAG start $(date -u +%FT%TZ) load[$L0] ===" | tee "$LOG"
TT_VISIBLE_DEVICES=$DEV TT_BIO_LEASE_CARDS=$DEV \
  TT_BIO_LEASE_HOLDER=worker:of3t-ditref \
  TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH="$WT" \
  "$PY" perf/of3t_diffusion/device_gradient.py "$@" >> "$LOG" 2>&1
RC=$?
E=$(date +%s)
L1=$(cut -d' ' -f1-3 /proc/loadavg)
kill $SIDE 2>/dev/null
CLKSTAT=$(awk '$2!="NA"{print $2}' "$CLK" | sort -n | awk '{a[n++]=$1; s+=$1} END{if(n==0){print "NA";exit} printf "n=%d min=%s med=%s max=%s mean=%.1f", n, a[0], a[int(n/2)], a[n-1], s/n}')
echo "=== arm $TAG exit $RC seconds $((E-S)) load_start[$L0] load_end[$L1] aiclk_during{$CLKSTAT} $(date -u +%FT%TZ) ===" | tee -a "$LOG"
printf '%s\n' "$TAG $RC $((E-S)) \"$L0\" \"$L1\" \"$CLKSTAT\"" >> "$SCR/ARMS.tsv"
exit $RC

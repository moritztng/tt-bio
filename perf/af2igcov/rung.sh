#!/bin/sh
# One af2ig coverage rung on pc's Blackhole p150a, with the clock it was measured at.
#
# On Blackhole the AICLK sets the fold time -- an idle card sits at 800 MHz and ramps when work
# lands -- so a reading taken before the launch describes nothing. The sampler here runs DURING
# the fold at 5 s and writes power and temperature beside the clock, which is also how a stall
# is told from a slow pass: a wedged card holds its idle power.
#
#   sh perf/af2igcov/rung.sh <tag> <yaml> <card>

# One place decides what a valid AICLK is: perf/lib/aiclk.sh, mirroring tt_bio.aiclk.
_L=$(cd "$(dirname "$0")" && pwd); . "${_L%/perf/*}/perf/lib/aiclk.sh" || exit 1
set -u
WT=/home/moritz/.coworker/wt/cov-below-bar-af2ig-bhp150a
PY=/home/moritz/tt-bio/env/bin/python3
TAG=$1; YAML=$2; DEV=${3:-0}
OUT=$WT/perf/af2igcov/out/$TAG
rm -rf "$OUT"; mkdir -p "$OUT"
SYS=/sys/class/tenstorrent/tenstorrent!$DEV
HW=$(ls -d "$SYS"/device/hwmon/hwmon*/ 2>/dev/null | head -1)

( while :; do
    printf '%s %s %s %s %s\n' "$(date +%s)" \
      "$(aiclk "$SYS")" \
      "$(awk '{printf "%.1f", $1/1000000}' "$HW/power1_input" 2>/dev/null || echo NA)" \
      "$(awk '{printf "%.1f", $1/1000}' "$HW/temp1_input" 2>/dev/null || echo NA)" \
      "$(cut -d' ' -f1 /proc/loadavg)" >> "$OUT/aiclk.log"
    sleep 5
  done ) &
SIDE=$!
trap 'kill "$SIDE" 2>/dev/null' EXIT INT TERM

cd "$WT" || exit 1
T0=$(date +%s)
TT_VISIBLE_DEVICES=$DEV TT_BIO_LEASE_CARDS=$DEV \
TT_BIO_LEASE_HOLDER=worker:cov-below-bar-af2ig-bhp150a \
TT_BIO_LEASE_TIMEOUT=2400 TT_METAL_LOGGER_LEVEL=FATAL \
  "$PY" -u -m tt_bio.main predict "$YAML" --model af2ig --accelerator tenstorrent \
    --device_ids "$DEV" --out_dir "$OUT/res" 2>&1 | tee "$OUT/rung.log"
RC=$?
T1=$(date +%s)
kill "$SIDE" 2>/dev/null
echo "$TAG rc=$RC wall_s=$((T1 - T0))" | tee "$OUT/wall.txt"

#!/usr/bin/env bash
# Sample Blackhole AICLK, power and temperature straight out of sysfs.
#
# tt-smi opens the device to read telemetry, which on a p300c is a real hazard: a fourth device open
# in a reset window wedges the chip, and every open brings up every visible card. tt-kmd already
# exports the same counters as plain files, so a sampler that only reads /sys costs nothing and
# cannot touch the run it is measuring.
#
#   clk_sampler.sh <out.csv> <hz>
set -u
OUT=${1:?out csv}; HZ=${2:-20}
SLEEP=$(python3 -c "print(1.0/$HZ)")
# hwmon index per chip: chip N sits on PCI 0000:0(N+1):00.0
declare -A HW
for h in /sys/class/hwmon/hwmon*; do
  [ "$(cat "$h/name" 2>/dev/null)" = blackhole ] || continue
  bdf=$(basename "$(readlink -f "$h/device")")
  case "$bdf" in 0000:01:00.0) HW[0]=$h;; 0000:02:00.0) HW[1]=$h;; 0000:03:00.0) HW[2]=$h;; 0000:04:00.0) HW[3]=$h;; esac
done
echo "t,chip,aiclk_mhz,power_uw,temp_mdegc,curr_ma" > "$OUT"
while :; do
  t=$(date +%s.%N)
  for n in 0 1 2 3; do
    c=$(cat "/sys/class/tenstorrent/tenstorrent!$n/tt_aiclk" 2>/dev/null || echo "")
    h=${HW[$n]:-}
    p=$(cat "$h/power1_input" 2>/dev/null || echo "")
    tm=$(cat "$h/temp1_input" 2>/dev/null || echo "")
    cu=$(cat "$h/curr1_input" 2>/dev/null || echo "")
    echo "$t,$n,$c,$p,$tm,$cu" >> "$OUT"
  done
  sleep "$SLEEP"
done

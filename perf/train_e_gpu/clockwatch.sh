#!/usr/bin/env bash
# Sample GPU clock, power and the compute-app list DURING a run, so a throttled or shared card
# cannot be mistaken for a slow one. Mirrors the AICLK discipline used for Blackhole numbers.
set -euo pipefail
OUT=${1:?out csv}
INT=${INT:-5}
echo "utc,sm_mhz,sm_max_mhz,mem_mhz,power_w,power_limit_w,temp_c,util_pct,throttle,n_apps,apps_mem_mib" > "$OUT"
while true; do
  read -r sm smax mem pw pl temp util <<<"$(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu --format=csv,noheader,nounits | tr -d ',')"
  thr=$(nvidia-smi -q -d PERFORMANCE 2>/dev/null | grep -A12 "Clocks Event Reasons" | grep -iE "Active" | grep -civ "Not Active" || true)
  apps=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | tr -d ' ')
  n=$(printf '%s\n' "$apps" | grep -c . || true)
  tot=$(printf '%s\n' "$apps" | paste -sd+ | bc 2>/dev/null || echo 0)
  echo "$(date -u +%FT%TZ),$sm,$smax,$mem,$pw,$pl,$temp,$util,$thr,$n,$tot" >> "$OUT"
  sleep "$INT"
done

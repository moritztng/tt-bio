#!/usr/bin/env bash
# Sample GPU clock, power, utilisation and the compute-app list DURING a run, not before.
# A perf number without a clock sampled during the measurement is not a measurement, and the
# compute-app list is the only way to see a co-tenant appear on the same physical card mid-run.
OUT=${1:?usage: clockwatch.sh <out.csv> [interval_s]}
INT=${2:-5}
echo "ts_utc,sm_mhz,mem_mhz,power_w,util_pct,mem_used_mib,temp_c,n_compute_apps" > "$OUT"
while true; do
  read -r sm mem pw util mu tmp <<<"$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits | tr -d ',')"
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  echo "$(date -u +%FT%TZ),$sm,$mem,$pw,$util,$mu,$tmp,$n" >> "$OUT"
  sleep "$INT"
done

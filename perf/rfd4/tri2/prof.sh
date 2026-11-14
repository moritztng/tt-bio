#!/bin/bash
# perf/rfd4/e2e/tri_prof.py under the device profiler (Tracy build tt-metal-b2z = ttnn 0.68.0), once per n.
#   prof.sh <out_dir> <card> <n...>
set -uo pipefail
wt=$(cd "$(dirname "$0")/../../.." && pwd)
out=$1; card=$2; shift 2
M=/home/ttuser/tt-metal-b2z
export TT_METAL_HOME=$M PYTHONPATH=$wt:$wt/perf:$M/ttnn:$M/tools:$M
unset LD_LIBRARY_PATH
export PATH=/home/ttuser/tt-bio-dev/env/bin:$PATH   # tracy -r relaunches the child as bare python3
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tri2
cd "$wt"
for n in "$@"; do
  raw=/dev/shm/rfd4-tri2-prof-$n; rm -rf "$raw"; mkdir -p "$out/$n"
  echo "n=$n start $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)"
  TT_METAL_PROFILER_DIR=$raw timeout 1500 python3 -m tracy -r --no-op-info-cache -o "$raw" --op-support-count 20000 -- \
    "$wt/perf/rfd4/e2e/tri_prof.py" --out "$out/$n/run.json" --n "$n" ${CALLS:+--calls $CALLS}
  echo "n=$n PROF EXIT $? $(date -u +%FT%TZ)"
  csv=$(ls -t "$raw"/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1)
  [ -n "$csv" ] && cp "$csv" "$out/$n/ops.csv" && python3 "$wt/perf/rfd4/e2e/tri_prof_table.py" "$out/$n/ops.csv" > "$out/$n/table.txt"
  rm -rf "$raw"
done
echo "end $(date -u +%FT%TZ)"

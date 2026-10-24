#!/bin/bash
# mm_sweep.py under the device profiler (Tracy build of the wheel's ttnn), then mm_table.py.
#   [SCRIPT=sdpa_sweep.py] mm_sweep.sh <out_dir> <card> [script args]
set -uo pipefail
wt=$(cd "$(dirname "$0")/../../.." && pwd)
out=$1; card=$2
M=/home/ttuser/tt-metal-b2z
raw=/dev/shm/rfd4-mmsweep-$$
rm -rf "$raw"; mkdir -p "$out"; out=$(cd "$out" && pwd)
cd "$wt"
export TT_METAL_HOME=$M PYTHONPATH=$wt:$wt/perf:$M/ttnn:$M/tools:$M TT_METAL_PROFILER_DIR=$raw
unset LD_LIBRARY_PATH
export PATH=/home/ttuser/tt-bio-dev/env/bin:$PATH
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tok512
echo "start $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)"
timeout 4800 python3 -m tracy -r --no-op-info-cache -o "$raw" --op-support-count 40000 -- \
  "$wt/perf/rfd4/tok512/${SCRIPT:-mm_sweep.py}" --out "$out/sweep.json" "${@:3}"
echo "PROF EXIT $? $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)"
csv=$(ls -t "$raw"/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1)
[ -n "$csv" ] && cp "$csv" "$out/ops.csv" && python3 "$wt/perf/rfd4/tok512/mm_table.py" "$out" > "$out/table.txt"
rm -rf "$raw"
echo "end $(date -u +%FT%TZ)"

#!/bin/bash
# call_prof.py under the device profiler on the Tracy build of the wheel's ttnn (as perf/rfd4/e2e/elt_prof.sh).
#   call_prof.sh <out_dir> <card> [call_prof.py args]      run from the tree being measured
set -uo pipefail
wt=$(cd "$(dirname "$0")/../../.." && pwd)
out=$(mkdir -p "$1" && cd "$1" && pwd); card=$2
M=/home/ttuser/tt-metal-b2z
raw=/dev/shm/rfd4-callprof-$$
rm -rf "$raw"; mkdir -p "$raw"
cd "$wt"
export TT_METAL_HOME=$M PYTHONPATH=$wt:$wt/perf:$M/ttnn:$M/tools:$M TT_METAL_PROFILER_DIR=$raw
unset LD_LIBRARY_PATH
export PATH=/home/ttuser/tt-bio-dev/env/bin:$PATH   # tracy -r relaunches the child as bare python3
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:rfd4-perf-p512}
echo "start $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)"
timeout 2400 python3 -m tracy -r --no-op-info-cache -o "$raw" --op-support-count 60000 -- \
  "$wt/perf/rfd4/p512/call_prof.py" --out "$out" "${@:3}"
echo "PROF EXIT $? $(date -u +%FT%TZ)"
csv=$(ls -t "$raw"/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1)
[ -n "$csv" ] && cp "$csv" "$out/ops.csv" && python3 "$wt/perf/rfd4/p512/call_table.py" "$out" > "$out/table.txt"
rm -rf "$raw"
echo "end $(date -u +%FT%TZ)"

#!/bin/bash
# call_prof.py under the device profiler on the Tracy build of the wheel's ttnn (tt-metal-b2z, v0.68.0).
#   call_prof.sh <out_dir> [card] [call_prof.py args]
set -uo pipefail
wt=$(cd "$(dirname "$0")/../../.." && pwd)
out=$1; card=${2:-3}
M=/home/ttuser/tt-metal-b2z
raw=/dev/shm/rfd4-callprof-$(basename "$wt")-$$
rm -rf "$raw"; mkdir -p "$out"
cd "$wt"
export TT_METAL_HOME=$M PYTHONPATH=$wt:$wt/perf:$M/ttnn:$M/tools:$M TT_METAL_PROFILER_DIR=$raw
unset LD_LIBRARY_PATH
export PATH=/home/ttuser/tt-bio-dev/env/bin:$PATH   # tracy -r relaunches the child as bare python3
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:$(basename "$wt")}
echo "start $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD) load $(cut -d' ' -f1 /proc/loadavg)"
timeout 1500 python3 -m tracy -r --no-op-info-cache -o "$raw" --op-support-count 40000 -- \
  "$wt/perf/rfd4/glue/call_prof.py" --out "$out/run.json" "${@:3}"
echo "PROF EXIT $? $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)"
csv=$(ls -t "$raw"/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1)
if [ -n "$csv" ]; then
  cp "$csv" "$out/ops.csv"
  python3 "$wt/perf/rfd4/e2e/elt_prof_table.py" "$out/ops.csv" --json "$out/table.json" > "$out/table.txt"
  python3 "$wt/perf/rfd4/glue/op_table.py" "$out" --call > "$out/ops_by_family.txt"
fi
rm -rf "$raw"
echo "end $(date -u +%FT%TZ)"

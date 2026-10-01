#!/bin/bash
# The shipped N=1 round under the device profiler, on a Tracy build of the wheel's own ttnn
# (tt-metal-b2z, v0.68.0, ENABLE_TRACY=ON). Kernel durations and op mix only; the wall is not.
#   prof.sh [rounds] [wait_pid]
set -uo pipefail
cd "$(dirname "$0")/../.."
r=${1:-4}
[ -n "${2:-}" ] && while kill -0 "$2" 2>/dev/null; do sleep 20; done
M=/home/ttuser/tt-metal-b2z
out=${PROF_OUT:-/dev/shm/bcp-roofline-prof}
tag=${PROF_TAG:-perf/bcp_roofline/out/prof_n1}
rm -rf "$out" "$tag"; mkdir -p "$tag"
export TT_METAL_HOME=$M PYTHONPATH=$PWD:/home/ttuser/.cache/bcp-roofline-pydeps:$M/ttnn:$M/tools:$M TT_METAL_PROFILER_DIR=$out
unset LD_LIBRARY_PATH
export PATH=/home/ttuser/bcx_e2e_venv/bin:$PATH   # tracy -r re-launches the child as bare python3
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$PWD/perf/bcp_roofline/out/xlacache}
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-0}
export TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:bcp-roofline}
echo "prof start $(date -u +%FT%TZ)"
timeout 2700 /home/ttuser/bcx_e2e_venv/bin/python3 -m tracy -r --no-op-info-cache \
  -o "$out" --op-support-count ${OPS:-60000} -- \
  "$PWD/perf/bcp_roofline/prof_round.py" --rounds "$r" --interleave 0 --trajectories 1 \
  --binder 146 --params /home/ttuser/bcx_e2e/af2_params --out "$PWD/$tag"
echo "PROF EXIT $? $(date -u +%FT%TZ)"
ls -la "$out" "$out"/reports/* 2>/dev/null | tail -20

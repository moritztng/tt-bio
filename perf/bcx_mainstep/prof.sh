#!/bin/bash
# The BC2 round under the device profiler, on main's own shipped ttnn wheel (0.68.0).
# Not a host self-time table: these are KERNEL durations off the device profiler.
cd "$(dirname "$0")/../.."
out=/dev/shm/bcx-mainstep-prof
rm -rf $out perf/bcx_mainstep/out/prof_s100
export PYTHONPATH=$PWD TT_METAL_PROFILER_DIR=$out
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-mainstep
timeout 2400 /home/ttuser/bcx_e2e_venv/bin/python3 -m tracy -r --no-op-info-cache \
  -o $out --op-support-count 600000 -- \
  $PWD/perf/bcx_round/run_round.py --rounds 3 --exact 0 \
  --out $PWD/perf/bcx_mainstep/out/prof_s100
echo "PROF EXIT $?"

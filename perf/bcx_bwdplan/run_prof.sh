#!/bin/bash
# bcx-bwdplan device census: base and lever arms in one process on the Tracy build (mar31,
# v0.69-dev, the build bcx-realcensus measured within 0.8 % of the wheel). Artifacts on tmpfs.
cd /home/ttuser/.coworker/wt/bcx-bwdplan
M=/home/ttuser/tt-metal-main-mar31
P=/dev/shm/bcx-bp-prof
export TT_METAL_HOME=$M PYTHONPATH=$M/ttnn:$M/tools:$M TT_METAL_PROFILER_DIR=$P
export TMPDIR=/dev/shm/bcx-bp-tmp OMP_NUM_THREADS=8
unset LD_LIBRARY_PATH
rm -rf $P
echo "== prof $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-bwdplan \
  timeout 3000 taskset -c 16-31 /home/ttuser/tt-bio-dev/env/bin/python3 -m tracy -r --no-op-info-cache \
  -o $P --op-support-count 30000 -- $PWD/perf/bcx_bwdplan/prof.py prof "$@"
echo "EXIT $? $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
/home/ttuser/tt-bio-dev/env/bin/python3 perf/bcx_bwdplan/prof.py sum --out devsum_n256.json "$@" \
  && rm -rf $P

#!/bin/bash
# The profiled census: one process on the Tracy build (mar31, v0.69-dev), artifacts on tmpfs.
cd /home/ttuser/.coworker/wt/bcx-realcensus
M=/home/ttuser/tt-metal-main-mar31
export TT_METAL_HOME=$M PYTHONPATH=$M/ttnn:$M/tools:$M TT_METAL_PROFILER_DIR=/dev/shm/bcx-realcensus-prof
unset LD_LIBRARY_PATH
rm -rf /dev/shm/bcx-realcensus-prof
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-realcensus \
  timeout 3000 /home/ttuser/tt-bio-dev/env/bin/python3 -m tracy -r --no-op-info-cache \
  -o /dev/shm/bcx-realcensus-prof --op-support-count 30000 -- \
  $PWD/perf/bcx_realcensus/realcensus.py prof --stacks evo,extra --ks 1,2 --warm 1 --steps 2 --roof-reps 3
echo "EXIT $?"

#!/bin/bash
# bcx-bytes device census: this row's arms in one process on the Tracy build (mar31, v0.69-dev,
# the build bcx-realcensus measured within 0.8 % of the wheel). qb1's root disk is full, so the
# profiler output, TMPDIR and the kernel cache live on tmpfs.
cd /home/ttuser/.coworker/wt/bcx-bytes
M=/home/ttuser/tt-metal-main-mar31
P=/dev/shm/bcx-by-prof
export TT_METAL_HOME=$M PYTHONPATH=$M/ttnn:$M/tools:$M TT_METAL_PROFILER_DIR=$P
export TMPDIR=/dev/shm/bcx-by-tmp TT_METAL_CACHE=/dev/shm/bcx-by-cache OMP_NUM_THREADS=8
unset LD_LIBRARY_PATH
rm -rf $P; mkdir -p $TMPDIR
OUT=${OUT:-prof_arms.json}
echo "== prof $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg) hb $(cat /sys/class/tenstorrent/tenstorrent\!3/tt_heartbeat)"
TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:bcx-bytes \
  timeout 3000 /home/ttuser/tt-bio-dev/env/bin/python3 -m tracy -r --no-op-info-cache \
  -o $P --op-support-count 30000 -- $PWD/perf/bcx_bytes/bytes.py arms --prof --out $OUT "$@"
echo "EXIT $? $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg) hb $(cat /sys/class/tenstorrent/tenstorrent\!3/tt_heartbeat)"
R=$(ls -t $P/reports/*/ops_perf_results_*.csv | head -1)
cp "$R" /dev/shm/bcx-by-${OUT%.json}.csv
/home/ttuser/tt-bio-dev/env/bin/python3 perf/bcx_bytes/bytes.py psum --report "$R" --out psum_${OUT} "$@"

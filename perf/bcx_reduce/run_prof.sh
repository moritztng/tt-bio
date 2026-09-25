#!/bin/bash
# bcx-reduce device census: this row's arms in one process on the Tracy build bcx-bytes used
# (mar31, v0.69-dev). qb1's root disk is full, so profiler output, TMPDIR and the kernel cache
# live on tmpfs. Card: logical 0 (PCI 01:00.0).
cd /home/ttuser/.coworker/wt/bcx-reduce
M=/home/ttuser/tt-metal-main-mar31
P=/dev/shm/bcx-rd-prof
HB=$(ls -d /sys/class/tenstorrent/*/ | while read d; do [ "$(basename $(readlink -f $d/device))" = 0000:01:00.0 ] && echo $d; done)
export TT_METAL_HOME=$M PYTHONPATH=$M/ttnn:$M/tools:$M TT_METAL_PROFILER_DIR=$P
export TMPDIR=/dev/shm/bcx-rd-tmp TT_METAL_CACHE=/dev/shm/bcx-rd-cache OMP_NUM_THREADS=8
unset LD_LIBRARY_PATH
rm -rf $P; mkdir -p $TMPDIR
OUT=${OUT:-prof_arms.json}
echo "== prof $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg) hb $(cat ${HB}tt_heartbeat)"
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-reduce \
  timeout 3000 /home/ttuser/tt-bio-dev/env/bin/python3 -m tracy -r --no-op-info-cache \
  -o $P --op-support-count 30000 -- $PWD/perf/bcx_reduce/ab.py arms --prof --out $OUT "$@"
echo "EXIT $? $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg) hb $(cat ${HB}tt_heartbeat)"
# tracy waits 15 s for its capture process to finish the host log and then gives up on the report.
# Under load a 700 MB log takes longer than that, so build the report from the saved logs here.
if ! ls $P/reports/*/ops_perf_results_*.csv >/dev/null 2>&1; then
  (cd $M/tools && /home/ttuser/tt-bio-dev/env/bin/python3 -c "from tracy import generate_report; from tracy.common import PROFILER_BIN_DIR; generate_report('$P', PROFILER_BIN_DIR, None, None)")
fi
R=$(ls -t $P/reports/*/ops_perf_results_*.csv | head -1)
cp "$R" /dev/shm/bcx-rd-${OUT%.json}.csv
/home/ttuser/tt-bio-dev/env/bin/python3 perf/bcx_bytes/bytes.py psum --report "$R" --out ../bcx_reduce/psum_${OUT} "$@"

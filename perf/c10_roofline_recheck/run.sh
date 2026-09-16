#!/usr/bin/env bash
set -euo pipefail
cd /home/ttuser/.coworker/wt/c10--roofline-recheck
export PATH=/home/ttuser/tt-bio-dev/env/bin:$PATH
export PYTHONPATH=$PWD:/home/ttuser/tt-metal-k10/ttnn:/home/ttuser/tt-metal-k10:/home/ttuser/tt-metal-k10/tools
export TT_METAL_HOME=/home/ttuser/tt-metal-k10
export TT_METAL_RUNTIME_ROOT=/home/ttuser/tt-metal-k10
export LD_LIBRARY_PATH=/home/ttuser/tt-metal-k10/build_Release/lib
export OMP_NUM_THREADS=2
export TT_BIO_AICLK=1350
export BENCHLOCK_WAIT_S=60
export BENCHLOCK_LOAD_WAIT_S=60
test "$(cat /sys/module/tenstorrent/srcversion)" = A10759A24565BC5BBE903C5
test "$(systemctl is-active qb2-endpoint-containment.service)" = active
case "${1:-}" in
  counter)
    /home/ttuser/.coworker/scripts/benchlock.sh c10--roofline-recheck -- env \
      TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:c10--roofline-recheck \
      python3 perf/c10_roofline_recheck/control.py --out perf/c10_roofline_recheck/counter
    ;;
  dm)
    /home/ttuser/.coworker/scripts/benchlock.sh c10--roofline-recheck -- env \
      TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:c10--roofline-recheck \
      python3 -m tracy -r -o perf/c10_roofline_recheck/tracy --enable-sum-profiling -- \
      perf/c10_roofline_recheck/calibrate_dm.py --out perf/c10_roofline_recheck/dm
    ;;
  *) echo "usage: bash perf/c10_roofline_recheck/run.sh counter|dm" >&2; exit 64;;
esac

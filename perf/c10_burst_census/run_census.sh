#!/usr/bin/env bash
set -euo pipefail
cd /home/ttuser/.coworker/wt/c10-burst-census
export PATH=/home/ttuser/tt-bio-dev/env/bin:$PATH
export PYTHONPATH=$PWD:/home/ttuser/tt-metal-k10/ttnn:/home/ttuser/tt-metal-k10:/home/ttuser/tt-metal-k10/tools
export TT_METAL_HOME=/home/ttuser/tt-metal-k10
export TT_METAL_RUNTIME_ROOT=/home/ttuser/tt-metal-k10
export LD_LIBRARY_PATH=/home/ttuser/tt-metal-k10/build_Release/lib
RUN_NAME=$(date -u +%Y%m%dT%H%M%SZ)
if [ "$#" -gt 0 ]; then RUN_NAME=$1; fi
[[ "$RUN_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid capture name" >&2; exit 64; }
OUT="$PWD/perf/c10_burst_census/runs/$RUN_NAME"
test ! -e "$OUT" || { echo "Capture already exists: $OUT" >&2; exit 64; }
mkdir -p "$OUT"
cp perf/c10_burst_census/census_criterion.json "$OUT/census_criterion.json"
export TMPDIR="$PWD/perf/c10_burst_census/tmp"
export OMP_NUM_THREADS=2
export TT_BIO_AICLK=1350
export BENCHLOCK_WAIT_S=60
export BENCHLOCK_LOAD_WAIT_S=60
test "$(cat /sys/module/tenstorrent/srcversion)" = A10759A24565BC5BBE903C5
test "$(systemctl is-active qb2-endpoint-containment.service)" = active
/home/ttuser/.coworker/scripts/benchlock.sh c10-burst-census -- env \
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:c10-burst-census \
  python3 -m tracy -r -p -o "$OUT/tracy" --enable-sum-profiling --dump-device-data-mid-run --disable-device-data-push-to-tracy --op-support-count 8192 -- \
  perf/c10_burst_census/census.py --out "$OUT/out"

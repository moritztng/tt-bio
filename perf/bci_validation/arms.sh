#!/bin/bash
# Issue #21 CPU arms. Serial on purpose: three folds at nice 10 beside the booth soak.
set -u
WT=/home/ttuser/.coworker/wt/bci-validation
PRE=/home/ttuser/bci_prefix
PY=/home/ttuser/fdv_fresh/venv/bin/python
OUT=/home/ttuser/bci_arms
mkdir -p $OUT
cd $WT
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
run () {   # name arm ttbio
  echo "=== $1  arm=$2  ttbio=$3  $(date -u +%H:%M:%SZ)"
  timeout 3600 nice -n 10 $PY perf/bci_validation/refold21.py \
      --arm "$2" --ttbio "$3" --out "$OUT/$1.json"
  echo "--- $1 rc=$? $(date -u +%H:%M:%SZ)"
}
run pure          pure    "$WT"
run spliced_prefix spliced "$PRE"
run spliced_fixed  spliced "$WT"
echo "ALLDONE $(date -u +%H:%M:%SZ)"

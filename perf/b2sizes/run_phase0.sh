#!/bin/bash
# Phase 0: which landed levers fire at 128 / 256 / 768 / 1024 aa, proven by the harness counters,
# plus an A/A pair per size. `on,on` is the shipped arm twice, so every number here is a floor or a
# census -- no lever is under test yet.
WT=/home/ttuser/.coworker/wt/boltz2-sizes-perf
cd $WT
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltz2-sizes-perf PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh boltz2-sizes-perf -- \
  $PY -u perf/other512/fold_ab_multi.py --model boltz2 --sizes "${1:-128,256}" \
      --arms on,on --out perf/b2sizes/phase0_"${2:-small}".json
echo "RC=$?"

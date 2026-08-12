#!/bin/bash
WT=/home/ttuser/.coworker/wt/boltz2-512aa-deep-perf
cd $WT
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltz2-512aa-deep-perf PYTHONPATH=$WT
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh boltz2-512aa-deep-perf -- \
  $PY -u perf/b2deep/decompose.py --model boltz2 --sizes 512 --deep \
      --arms on,on --out perf/b2deep/decomp_512.json
echo "RC=$?"

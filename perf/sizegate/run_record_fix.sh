#!/bin/bash
WT=/home/ttuser/.coworker/wt/protenix-v1-sizeladder-baseline
cd $WT
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:protenix-v1-sizeladder-baseline
exec /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/release_gate.py --model size-ladder \
    --size-ladder-record --size-ladder-models nesso1,openfold3 \
    > $WT/perf/sizegate/record_p300c_refix.log 2>&1

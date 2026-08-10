#!/bin/sh
# Wait for the 512 aa A/B to release card 3, then take the 298 aa byte-identity arm and the 384 aa null.
cd /home/ttuser/.coworker/wt/protenix-trunk--z-rowblock || exit 1
while pgrep -f "fold_ab512.py --sizes 512" >/dev/null 2>&1; do sleep 10; done
export TT_VISIBLE_DEVICES=3
export TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-rowblock
export PYTHONPATH=/home/ttuser/.coworker/wt/protenix-trunk--z-rowblock
exec /home/ttuser/tt-bio-dev/env/bin/python3 perf/size512/fold_ab512.py \
    --sizes 298,384 --arms on,rb_fit \
    --fixdir perf/size512/fixtures \
    --out perf/rowblock/fold_ab_298_384_qb1c3.json --host qb1 --chip 3

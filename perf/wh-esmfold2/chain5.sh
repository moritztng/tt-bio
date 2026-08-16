#!/bin/bash
# wh-perf-esmfold2 p4: the Wormhole fold-level A/B. Arms base,A only -- lever C is an evidenced
# NO-GO at 320 (L1 arm 27.5 % slower) and OOMs outright at 512, so putting it on the box would
# spend a shared production machine measuring a known debit. `base` repeated per round is the
# A/A floor every delta is judged against.
cd /home/cust-team/mthuening/whbase/wt-esmfold2 || exit 1
PY=/home/cust-team/mthuening/whbase/tt-bio/env/bin/python
export TT_VISIBLE_DEVICES=30
export TT_BIO_LEASE_HOLDER=worker:wh-perf-esmfold2
export TT_METAL_LOGGER_LEVEL=FATAL
export BENCHLOCK_FILE=/home/cust-team/mthuening/whbase/benchlock
export BENCHLOCK_FOREIGN_RE="xmodel_ab|wh_esm|decomp\.py|fold_ab|screen_wh|roofs\.py"
export BENCHLOCK_MAXLOAD=20
export BENCHLOCK_LOAD_WAIT_S=120
BL=/home/cust-team/mthuening/whbase/benchlock.sh
O=perf/wh-esmfold2/out
echo "=== fold A/B 512 start $(date -u +%H:%M:%S) ==="
$BL wh-perf-esmfold2 -- $PY -u perf/wh-esmfold2/fold_ab512.py \
    --model esmfold2 --size 512 --fast --arms base,A --rounds 3 \
    --out "$O/fold_ab_512_wh.json" > "$O/fold_ab_512_wh.log" 2>&1
echo "=== fold A/B 512 rc=$? $(date -u +%H:%M:%S) ==="
echo "=== chain5 done $(date -u +%H:%M:%S) ==="

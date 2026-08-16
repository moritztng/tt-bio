#!/bin/bash
# wh-perf-esmfold2 p5: the Blackhole neutrality A/B re-run, --fast, AFTER 7f9335fc guarded the
# small-grid fc1 dtype on _IS_SMALL_GRID. The pre-fix run is kept as fold_ab_512_qb1c1_fast.json;
# this writes the _fixed file so the defect and its repair are both on the record.
cd /home/ttuser/.coworker/wt/wh-perf-esmfold2 || exit 1
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:wh-perf-esmfold2
export TT_METAL_LOGGER_LEVEL=FATAL
BL=/home/ttuser/.coworker/scripts/benchlock.sh
O=perf/wh-esmfold2/out
echo "=== bh neutrality fast REFIX start $(date -u +%H:%M:%S) ==="
$BL wh-perf-esmfold2 -- python3 -u perf/wh-esmfold2/fold_ab512.py \
    --model esmfold2 --size 512 --fast --arms base,A --rounds 2 \
    --out "$O/fold_ab_512_qb1c1_fast_fixed.json" > "$O/fold_ab_512_qb1c1_fast_fixed.log" 2>&1
echo "=== bh neutrality fast REFIX rc=$? $(date -u +%H:%M:%S) ==="
echo "=== chain_bh3 done $(date -u +%H:%M:%S) ==="

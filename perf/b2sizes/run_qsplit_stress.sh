#!/bin/bash
# OOM-stress + parity evidence for the triatt q-split (TT_BIO_TRIATT_MASK_Q_SPLIT) at the two sizes
# it is proven at. Reference arms run first (the L1-refusal memos are process-global, cleared per
# arm by set_arm), then the lever arms repeat back to back: a leak or a peak-only overflow shows up
# on repeat two or three, not one. Per-arm CIF sha256 is the parity instrument (plDDT is None on
# this model and the fixtures carry artificial hinges, so RMSD is unreadable at these sizes).
#
# Co-tenanted timings are indicative only; run under benchlock for a timing claim.
WT=/home/ttuser/.coworker/wt/boltz2-qparallel-768-1024-land
cd $WT || exit 70
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:boltz2-qparallel-768-1024-land PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3

echo "=== 1024 aa stress $(date -Is) ==="
$PY -u perf/other512/fold_ab_multi.py --model boltz2 --sizes 1024 \
    --arms on,on,qsplit,qsplit,qsplit --out perf/b2sizes/qsplit_stress_1024_qb1.json
echo "1024 RC=$?"

echo "=== 768 aa confirm $(date -Is) ==="
$PY -u perf/other512/fold_ab_multi.py --model boltz2 --sizes 768 \
    --arms on,on,qsplit --out perf/b2sizes/qsplit_confirm_768_qb1.json
echo "768 RC=$?"
echo "=== done $(date -Is) ==="

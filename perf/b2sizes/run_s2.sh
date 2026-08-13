#!/bin/bash
# S2: K2 at 768/1024 aa. The gate that refuses every call there is `q_per_core == 1`, and the reason
# it fails is that K2 asks for `q_pf = 1`, so `q_per_core == q_num_chunks`. Above ~640 aa the widest
# q_chunk overflows L1 and the fold runs 2 chunks. `_Q_PARALLEL` asks for `q_pf = q_num_chunks`.
#
# Step 2b is the off-fold screen at the production shapes. Pre-committed kill rule from
# state/boltz2-sizes-perf.md S2: under 1.10x at (768,4,768,32), or torch.equal false, is NO-GO and
# the fold A/B does not run. Predicted fold landing, priced at the screen's own measured per-call
# delta and at nothing else: 560 x (ms_A - ms_B) / 1000 seconds per fold at each size.
WT=/home/ttuser/.coworker/wt/boltz2-sizes-perf
cd $WT || exit 70
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltz2-sizes-perf PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh

echo "=== S2b screen $(date -Is) ==="
$BL boltz2-sizes-perf -- $PY -u perf/b2sizes/s2b_mask_q_parallel.py \
    --sizes 768,1024 --reps 7 --out perf/b2sizes/s2b_screen.json
echo "screen RC=$?"

VERDICT=$($PY perf/b2sizes/s2b_verdict.py)
echo "S2b VERDICT: $VERDICT"
case "$VERDICT" in
  GO*) ;;
  *) echo "NO-GO on the pre-committed kill rule; no fold A/B. Done."; exit 0 ;;
esac

echo "=== S2 fold A/B 768 $(date -Is) ==="
$BL boltz2-sizes-perf -- $PY -u perf/other512/fold_ab_multi.py --model boltz2 --sizes 768 \
    --arms on,on,nok2,k2q,on --out perf/b2sizes/s2_ab_768.json
echo "ab768 RC=$?"

echo "=== S2 fold A/B 1024 $(date -Is) ==="
$BL boltz2-sizes-perf -- $PY -u perf/other512/fold_ab_multi.py --model boltz2 --sizes 1024 \
    --arms on,on,k2q,on --out perf/b2sizes/s2_ab_1024.json
echo "ab1024 RC=$?"
echo "=== S2 done $(date -Is) ==="

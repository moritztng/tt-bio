#!/usr/bin/env bash
# Steps 3-5 of the plan, in ONE benchlock hold, cheapest-and-most-gating first so a truncated
# run still leaves the valuable rows. Sequential with `;` not `&&`: a predicted CB clash at
# (1024, rows=32) IS a result and must not cancel what follows.
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-sizes-perf
cd "$WT" || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
ENVP="env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:esmfold2-sizes-perf PYTHONPATH=$WT"
O=perf/esm2sizes

echo "=== campaign4 start $(date -Is) loadavg $(cut -d' ' -f1-3 /proc/loadavg) ==="

echo "--- STEP 4a: THE SCREEN at 768, rows 8/16/32 (768 is where the shipped window ends) ---"
$ENVP $PY perf/esm2land/probe_parity.py --L 768 --rows 8 16 32 --out $O/screen_768_c2.json 2>&1
echo "RC_SCREEN768=$?"

echo "--- STEP 4b: THE SCREEN at 1024, rows 8/16/32 (rows=32 predicted to throw a CB clash) ---"
$ENVP $PY perf/esm2land/probe_parity.py --L 1024 --rows 8 16 32 --out $O/screen_1024_c2.json 2>&1
echo "RC_SCREEN1024=$?"

echo "--- STEP 3: roofs on card 2 at 768 and 1024 ---"
$ENVP $PY perf/esm512/roofs.py --L 768  --out $O/roofs_768_c2.json 2>&1
echo "RC_ROOF768=$?"
$ENVP $PY perf/esm512/roofs.py --L 1024 --out $O/roofs_1024_c2.json 2>&1
echo "RC_ROOF1024=$?"

echo "--- STEP 5: the 768 fold A/B. armAC IS WHAT SHIPS AT 768; armA isolates the row block. ---"
$ENVP $PY perf/esm2land/fold_ab.py --size 768 --arms base,armA,armAC --rounds 2 \
    --out $O/fold_768_c2.json 2>&1
echo "RC_FOLDAB768=$?"

echo "=== campaign4 end $(date -Is) ==="

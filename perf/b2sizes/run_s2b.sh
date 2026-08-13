#!/bin/bash
# S2 execution chain, relaunch 2026-08-13T02:2xZ. Supersedes run_s2.sh, which was killed while it
# still had 20 minutes left on a 5400 s benchlock wait it was going to lose (protenix-v2-sizes-perf
# has held the box since 01:08 for a 1024 aa fold whose cold arm alone is 315 s). Two changes:
#
#   1. BENCHLOCK_WAIT_S=21600, so the chain survives the queue instead of timing out at 02:38 and
#      reporting a NOFILE verdict that reads like a failed screen.
#   2. The fold A/B at 768 and 1024 now runs on BOTH branches of the S2b kill rule. The k2q arm is
#      what the kill rule gates; nok2 and nok1 are not. `nok2` at 768 is the census integrity check
#      that state doc S2 step 2a asked for -- K2 already declines all 1120 calls there, so nok2 MUST
#      read identical to `on` within the A/A floor or the counters are lying -- and `nok1` prices
#      what K1 is still buying at the two large sizes, where it serves 560/560.
#
# Screen kill rule, pre-committed, applied by s2b_verdict.py and not by a human reading the JSON:
# under 1.10x at (768,4,768,32) off-fold, or torch.equal false, is NO-GO for k2q.
WT=/home/ttuser/.coworker/wt/boltz2-sizes-perf
cd $WT || exit 70
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltz2-sizes-perf PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_WAIT_S=21600
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh

echo "=== S2b screen $(date -Is) ==="
$BL boltz2-sizes-perf -- $PY -u perf/b2sizes/s2b_mask_q_parallel.py \
    --sizes 768,1024 --reps 7 --out perf/b2sizes/s2b_screen.json
echo "screen RC=$?"

VERDICT=$($PY perf/b2sizes/s2b_verdict.py)
echo "S2b VERDICT: $VERDICT"
case "$VERDICT" in
  GO*) A768=on,on,nok2,k2q,on ; A1024=on,on,k2q,on ;;
  *)   A768=on,on,nok2,nok1,on ; A1024=on,on,nok1,on ;;
esac
echo "arms 768=$A768  1024=$A1024"

echo "=== S2 fold A/B 768 $(date -Is) ==="
$BL boltz2-sizes-perf -- $PY -u perf/other512/fold_ab_multi.py --model boltz2 --sizes 768 \
    --arms $A768 --out perf/b2sizes/s2_ab_768.json
echo "ab768 RC=$?"

echo "=== S2 fold A/B 1024 $(date -Is) ==="
$BL boltz2-sizes-perf -- $PY -u perf/other512/fold_ab_multi.py --model boltz2 --sizes 1024 \
    --arms $A1024 --out perf/b2sizes/s2_ab_1024.json
echo "ab1024 RC=$?"
echo "=== S2 done $(date -Is) ==="

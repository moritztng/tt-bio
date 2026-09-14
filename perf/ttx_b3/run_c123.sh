#!/usr/bin/env bash
# B3 close-out, three parallel legs on three cards. These are CENSUS/CORRECTNESS legs, not the
# timing claim: they run concurrently, so no wall-clock number from them is quotable. The shipped
# timing claim comes from perf/ttx_b3/fold_ab_esm512_c0.json, which ran alone under benchlock.
set -u
WT=/home/ttuser/.coworker/wt/ttx-b3-pairffn-fc1-l1-ship
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export TT_BIO_LEASE_HOLDER=worker:ttx-b3-pairffn-fc1-l1-ship PYTHONPATH="$WT"

# card 1 (the grant): does the L1 destination actually LAND at 512 aa, by the latch census?
( TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
  "$PY" -u perf/esm3p4land/fold_ab.py --model esmfold2 --size 512 --rounds 1 --arms base,l2 \
    --out perf/ttx_b3/fold_ab_esm512_latch_c1.json ) >perf/ttx_b3/latch_c1.log 2>&1 &
P1=$!

# card 2 (fanout): does Boltz-2 reach the gated call site at all?
( TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=1,2 \
  "$PY" -u perf/esm3p4land/fold_ab.py --model boltz2 --size 512 --rounds 1 --arms base,l2 \
    --out perf/ttx_b3/fold_ab_b2_512_c2.json ) >perf/ttx_b3/b2_c2.log 2>&1 &
P2=$!

# card 3 (fanout): the release gate's own ESMFold2 legs on this branch.
( TT_BIO_LEASE_CARDS=1,3 \
  "$PY" -u scripts/full_parity_gate.py --workers tt-quietbox2:3 \
    --leg esmfold2-trpcage --leg esmfold2-fast-trpcage --leg esmfold2-cocrystal \
    --workdir /home/ttuser/.coworker/wt/ttx-b3-pairffn-fc1-l1-ship/perf/ttx_b3/gate_work \
    --out perf/ttx_b3/gate_esmfold2_c3.json --load-ceiling 64 ) \
  >perf/ttx_b3/gate_c3.log 2>&1 &
P3=$!

wait $P1; echo "latch_c1 rc=$?"
wait $P2; echo "b2_c2 rc=$?"
wait $P3; echo "gate_c3 rc=$?"
echo "=== ALL DONE $(date -Is) ==="

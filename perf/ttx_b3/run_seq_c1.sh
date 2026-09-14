#!/usr/bin/env bash
# B3 close-out, SEQUENTIAL on the granted card 1. The earlier three-card parallel attempt
# (perf/ttx_b3/run_c123.sh) died when qb2 rebooted at 14:11; the host had already rebooted twice
# today (08:28, 09:25), so this stays on one card and one process at a time.
set -u
WT=/home/ttuser/.coworker/wt/ttx-b3-pairffn-fc1-l1-ship
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
       TT_BIO_LEASE_HOLDER=worker:ttx-b3-pairffn-fc1-l1-ship PYTHONPATH="$WT"

echo "=== latch512 start $(date -Is) ==="
"$PY" -u perf/esm3p4land/fold_ab.py --model esmfold2 --size 512 --rounds 1 --arms base,l2 \
  --out perf/ttx_b3/fold_ab_esm512_latch_c1.json
echo "=== latch512 rc=$? end $(date -Is) ==="

echo "=== boltz2 start $(date -Is) ==="
"$PY" -u perf/esm3p4land/fold_ab.py --model boltz2 --size 512 --rounds 0 --arms base,l2 \
  --out perf/ttx_b3/fold_ab_b2_512_c1.json
echo "=== boltz2 rc=$? end $(date -Is) ==="

echo "=== gate start $(date -Is) ==="
"$PY" -u scripts/full_parity_gate.py --workers tt-quietbox2:1 \
  --leg esmfold2-trpcage --leg esmfold2-fast-trpcage --leg esmfold2-cocrystal \
  --workdir "$WT/perf/ttx_b3/gate_work" --out perf/ttx_b3/gate_esmfold2_c1.json
echo "=== gate rc=$? end $(date -Is) ==="
echo "=== SEQ DONE $(date -Is) ==="

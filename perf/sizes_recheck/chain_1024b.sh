#!/usr/bin/env bash
# Continuation of chain_1024.sh with a 1500 s per-process timeout: 900 s was too tight for a
# protenix 1024 process, which pays a ~370 s cold fold before its single timed arm.
set -u
WT=/home/ttuser/.coworker/wt/sizes-recheck-protenix-openfold3
cd "$WT" || exit 1
OUT=$WT/perf/sizes_recheck
BL=/home/ttuser/.coworker/scripts/benchlock.sh
PRE="TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:sizes-recheck-protenix-openfold3 PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm"

run() {
  local tag="$1"; shift
  echo "=== $tag START $(date -Is) ==="
  BENCHLOCK_WAIT_S=3600 BENCHLOCK_LOAD_WAIT_S=600 \
    "$BL" worker:sizes-recheck-protenix-openfold3 -- \
    env $PRE OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt \
    timeout -k 20 1500 python3 -u "$@"
  local rc=$?
  echo "=== $tag RC=$rc $(date -Is) ==="
  if [ $rc -ne 0 ]; then
    echo "=== $tag nonzero RC, resetting card 1 ==="
    ~/.local/bin/tt-smi -r 1 >/dev/null 2>&1
    sleep 20
  fi
  return 0
}

# on_a is retried only if the 900 s process that is running now did not produce it.
if [ ! -f $OUT/px_1024_on_a_qb1c1.json ]; then
  run px1024_on_a perf/size512/fold_ab512.py --model protenix-v2 --sizes 1024 --arms on --out $OUT/px_1024_on_a_qb1c1.json
fi
run px1024_base4 perf/size512/fold_ab512.py --model protenix-v2 --sizes 1024 --arms base4 --out $OUT/px_1024_base4_qb1c1.json
run of3_1024_b   perf/other512/fold_ab_multi.py --model openfold3 --sizes 1024 --arms on   --out $OUT/of3_1024_qb1c1_b.json
run px1024_on_b  perf/size512/fold_ab512.py --model protenix-v2 --sizes 1024 --arms on     --out $OUT/px_1024_on_b_qb1c1.json
echo "=== CHAIN2 DONE $(date -Is) ==="

#!/usr/bin/env bash
# The three rungs owed by section 15, qb1 card 1, one arm per process (13.1 stall).
WT=/home/ttuser/.coworker/wt/sizes-recheck-protenix-openfold3
cd "$WT" || exit 1
OUT=$WT/perf/sizes_recheck
BL=/home/ttuser/.coworker/scripts/benchlock.sh
PRE="TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:sizes-recheck-protenix-openfold3 PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm"

step () {  # name  timeout  cmd...
  local name=$1 tmo=$2; shift 2
  echo "=== START $name $(date -Is)"
  env $PRE OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt \
    "$BL" sizes-recheck-protenix-openfold3 -- timeout -k 20 "$tmo" "$@"
  local rc=$?
  echo "=== END $name rc=$rc $(date -Is)"
  if [ $rc -ne 0 ]; then
    echo "=== resetting card 1 after rc=$rc"
    ~/.local/bin/tt-smi -r 1 >/dev/null 2>&1; echo "=== reset rc=$?"
  fi
  return $rc
}

step of3_768_a 900 python3 -u perf/other512/fold_ab_multi.py --model openfold3 --sizes 768 --arms on --out $OUT/of3_768_qb1c1_a.json
step of3_768_b 900 python3 -u perf/other512/fold_ab_multi.py --model openfold3 --sizes 768 --arms on --out $OUT/of3_768_qb1c1_b.json
step of3_1024_a 1800 python3 -u perf/other512/fold_ab_multi.py --model openfold3 --sizes 1024 --arms on --out $OUT/of3_1024_qb1c1_a.json
step px_1024_on_a 1300 python3 -u perf/size512/fold_ab512.py --model protenix-v2 --sizes 1024 --arms on --out $OUT/px_1024_on_a_qb1c1.json
step px_1024_base4 1300 python3 -u perf/size512/fold_ab512.py --model protenix-v2 --sizes 1024 --arms base4 --out $OUT/px_1024_base4_qb1c1.json
step px_1024_on_b 1300 python3 -u perf/size512/fold_ab512.py --model protenix-v2 --sizes 1024 --arms on --out $OUT/px_1024_on_b_qb1c1.json
step of3_1024_b 1800 python3 -u perf/other512/fold_ab_multi.py --model openfold3 --sizes 1024 --arms on --out $OUT/of3_1024_qb1c1_b.json
echo "=== ALL DONE $(date -Is)"

#!/usr/bin/env bash
# Pass-2 driver, qb2 card 1, ttnn 0.68.0, 11x10. Two jobs the pass-1 ladder could not deliver:
#   1. the 768 aa q-split repro -- pass 1 wedged on exactly this arm, and q-split is the SHIPPED
#      default (_Q_SPLIT=1, _Q_SPLIT_MAX_S=1024), so a reproducible hang here is a user-facing bug.
#   2. the ESMFold2 ladder ONE PROCESS PER SIZE. The single-process ladder OOMs at the 256->512
#      hop because build_fold reloads ESMC-6B without freeing the resident copy
#      (bank_manager.cpp:439, 4276706304 of 4278190016 B per bank already allocated).
set -u
WT=/home/ttuser/.coworker/wt/sizes-recheck-boltz2-esmfold2
PY=/home/ttuser/tt-bio-dev/env/bin/python3
SMI=/home/ttuser/.tenstorrent-venv/bin/tt-smi
O=$WT/perf/sizesrecheck
cd "$WT" || exit 1
E="env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:sizes-recheck-boltz2-esmfold2 PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm"

echo "=== $(date -u +%H:%M:%SZ) job1 boltz2 768 qsplit repro ==="
timeout -s KILL 480 $E "$PY" perf/other512/fold_ab_multi.py --model boltz2 \
    --sizes 768 --arms qsplit,qsplit --out "$O/repro_b2_768_qsplit.json" > "$O/repro_b2_768_qsplit.log" 2>&1
echo "job1 rc=$? at $(date -u +%H:%M:%SZ)"
"$SMI" -r 1 >/dev/null 2>&1; sleep 5

for S in 512 768 1024; do
  echo "=== $(date -u +%H:%M:%SZ) job esmfold2 $S ==="
  timeout -s KILL 1500 $E "$PY" perf/other512/fold_ab_multi.py --model esmfold2 \
      --sizes "$S" --arms on,on,on --out "$O/ef2_${S}_qb2c1.json" > "$O/ef2_${S}_qb2c1.log" 2>&1
  echo "esmfold2 $S rc=$? at $(date -u +%H:%M:%SZ)"
done
echo "=== $(date -u +%H:%M:%SZ) driver done ==="

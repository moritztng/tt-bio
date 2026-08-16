#!/bin/bash
# wh-perf-esmfold2 p3: the Blackhole arms. qb1 card 1 (13x10 = 130 cores).
#   1. the --fast 512 aa arm the audit left owed -- it turns the empty ESMFold2 gap cell
#      into a real architecture number instead of a precision-mixed 2.94x.
#   2. the non-fast shipped default, as the Blackhole neutrality reference.
cd /home/ttuser/.coworker/wt/wh-perf-esmfold2 || exit 1
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:wh-perf-esmfold2
export TT_METAL_LOGGER_LEVEL=FATAL
export ESM_ROOT=/home/ttuser/esm
export PYTHONPATH=/home/ttuser/.coworker/wt/wh-perf-esmfold2
BL=/home/ttuser/.coworker/scripts/benchlock.sh
O=perf/wh-esmfold2/out
mkdir -p "$O"
for arm in fast plain; do
  extra=""; [ "$arm" = fast ] && extra="--fast"
  echo "=== bh $arm start $(date -u +%H:%M:%S) ==="
  $BL wh-perf-esmfold2 -- python3 -u perf/of3_4xpd/xmodel_ab.py --model esmfold2 --size 512 \
      $extra --tree "$PWD" --repeat 3 --out "$O/base_512_qb1c1_${arm}.json" \
      > "$O/base_512_qb1c1_${arm}.log" 2>&1
  echo "=== bh $arm rc=$? $(date -u +%H:%M:%S) ==="
done
echo "=== chain_bh done $(date -u +%H:%M:%S) ==="

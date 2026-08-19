#!/usr/bin/env bash
# Ladder driver for sizes-recheck-boltz2-esmfold2. qb2 card 1, ttnn 0.68.0, 11x10 grid --
# the SAME host/card/ttnn as boltz2-sizes-perf 2.1 (08-13) and the esmfold2 08-14 anchor,
# so the Phase 2 comparison is same-config, not just same-ratio.
set -u
WT=/home/ttuser/.coworker/wt/sizes-recheck-boltz2-esmfold2
PY=/home/ttuser/tt-bio-dev/env/bin/python3
MODEL="$1"; SIZES="$2"; TAG="$3"; ARMS="${4:-on,on,on}"
cd "$WT" || exit 1
exec /home/ttuser/.coworker/scripts/benchlock.sh worker:sizes-recheck-boltz2-esmfold2 -- \
  env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:sizes-recheck-boltz2-esmfold2 \
      PYTHONPATH="$WT" ESM_ROOT=/home/ttuser/esm \
  "$PY" perf/other512/fold_ab_multi.py --model "$MODEL" \
      --sizes "$SIZES" --arms "$ARMS" \
      --out "perf/sizesrecheck/ladder_${TAG}.json"

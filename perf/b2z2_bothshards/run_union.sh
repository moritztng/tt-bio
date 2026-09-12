#!/usr/bin/env bash
# Wait for the p300c pair (qb2 chips 2+3) to be free, take benchlock, run the four-arm union
# session. Cards 2+3 are this row's lease but a sibling row fanned a size ladder onto them at
# 23:21 UTC, so the wait is real and not a formality.
set -u
WT=/home/ttuser/.coworker/wt/b2z2-p300c-both-shards
cd "$WT" || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=perf/b2z2_bothshards/b2z2_union_mesh2.json
ARMS="base,trunk,atom,both,base,trunk,atom,both,base,trunk,atom,both,base,trunk,atom,both,base,trunk,atom,both,base,trunk,atom,both"

free_pair() {
  for d in 2 3; do
    sudo lsof "/dev/tenstorrent/$d" >/dev/null 2>&1 && return 1
  done
  return 0
}

for i in $(seq 1 60); do
  if free_pair; then echo "[$(date -Is)] pair 2+3 free after ${i} checks"; break; fi
  echo "[$(date -Is)] waiting: pair 2+3 still held"
  sleep 20
done
free_pair || { echo "pair never freed"; exit 75; }

export TT_VISIBLE_DEVICES=2,3
export TT_BIO_LEASE_CARDS=2,3
export TT_BIO_LEASE_HOLDER=worker:b2z2-p300c-both-shards
export BENCHLOCK_WAIT_S=900
exec /home/ttuser/.coworker/scripts/benchlock.sh worker:b2z2-p300c-both-shards -- \
  "$PY" perf/b2z2_bothshards/union_fold.py \
     --arms "$ARMS" --mesh 2 --trace 1 --tag u2 --out "$OUT"

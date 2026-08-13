#!/bin/bash
# Hang-bisect probe (ws:boltz2-qb2-hang-bisect): one size, given arms, hard timeout.
# timeout sends TERM at TMO, KILL at TMO+30 (native busy-poll ignores TERM).
# After a timeout the card MUST be reset (tt-smi -r 2) and verified before the next leg.
WT=/home/ttuser/.coworker/wt/boltz2-qb2-hang-bisect
cd "$WT" || exit 70
PY=/home/ttuser/tt-bio-dev/env/bin/python3
SIZE=$1; ARMS=$2; OUT=$3; TMO=${4:-900}
echo "=== ${SIZE} aa arms=${ARMS} tmo=${TMO} $(date -Is) ==="
timeout -k 30 "$TMO" env TT_VISIBLE_DEVICES=${CARD:-2} TT_BIO_LEASE_HOLDER=worker:boltz2-qb2-hang-bisect \
  PYTHONPATH="$WT" ESM_ROOT=/home/ttuser/esm "$PY" -u \
  perf/other512/fold_ab_multi.py --model boltz2 --sizes "$SIZE" --arms "$ARMS" --out "$OUT"
RC=$?
echo "RC=$RC at $(date -Is)"
exit $RC

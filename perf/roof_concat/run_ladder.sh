#!/usr/bin/env bash
# Retry the ladder until a card lease is actually free. qb2's four cards are leased by other
# workers' chains and turn over as they finish, so "card 0 is your grant" and "card 0 is open
# right now" are different facts; the device lease refuses the second one, correctly.
set -u
WT=/home/ttuser/.coworker/wt/roof-concat-heads-bh
cd "$WT" || exit 1
OUT="$1"; shift
LOG="$1"; shift
for i in $(seq 1 40); do
  echo "=== attempt $i $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg)" >> "$LOG"
  BENCHLOCK_LOAD_WAIT_S=120 BENCHLOCK_WAIT_S=900 \
    bash "$HOME/.coworker/scripts/benchlock.sh" roof-concat-heads-bh -- \
    env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
        TT_BIO_LEASE_HOLDER=worker:roof-concat-heads-bh PYTHONPATH="$WT" \
    timeout 1500 /home/ttuser/tt-bio-dev/env/bin/python "$@" --out "$OUT" >> "$LOG" 2>&1
  rc=$?
  echo "=== attempt $i rc=$rc" >> "$LOG"
  [ -s "$OUT" ] && { echo "=== DONE $(date -Is)" >> "$LOG"; exit 0; }
  sleep 20
done
echo "=== GAVE UP $(date -Is)" >> "$LOG"
exit 1

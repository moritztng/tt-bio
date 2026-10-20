#!/bin/bash
# Run chain1.sh on CARD once the card is genuinely idle: no process holds /dev/tenstorrent/$CARD and no
# shipflow leg is queued on its flock. Takes the same flock for the chain, so a later shipflow leg waits
# behind it instead of opening the card under it. Gives up after MAXH hours.
#   CARD=1 setsid nohup bash perf/bc4_elem/wait_run.sh > _bc4out/wait_run.log 2>&1 &
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
LOCK=$HOME/spd_qb2_chip$CARD.lock; END=$(( $(date +%s) + ${MAXH:-10}*3600 ))
held() { for f in /proc/[0-9]*/fd/*; do [ "$(readlink "$f" 2>/dev/null)" = /dev/tenstorrent/$CARD ] && return 0; done; return 1; }
while [ "$(date +%s)" -lt "$END" ]; do
  if ! held && ! pgrep -f "spd_qb2_chip$CARD.lock" > /dev/null; then
    echo "$(date -u +%FT%TZ) card $CARD idle, starting chain1"
    exec flock "$LOCK" bash perf/bc4_elem/chain1.sh
  fi
  sleep 60
done
echo "$(date -u +%FT%TZ) gave up after ${MAXH:-10} h"

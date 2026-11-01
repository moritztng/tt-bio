#!/bin/bash
# Runs on qb1: when the current qb1_fanout.sh exits (its book.py windows ended), books cards 0 and 2 again for
# SECONDS (default 6 h; book.py starts them once the cards are clear) and relaunches the fanout, which skips every
# rc=0 fold. ~/mol1-quality/EXTEND exists while this waits, so qb2 sync_qb1.sh does not read the old FANOUT-EXIT as the end.
set -uo pipefail
cd ~/mol1-quality; S=${1:-21600}; say() { echo "$(date -u +%FT%TZ) $*" >> extend.log; }
touch EXTEND; say "waiting for FANOUT-EXIT (pid $$)"
until grep -q FANOUT-EXIT fanout.log 2>/dev/null; do sleep 60; done
mv fanout.log fanout.log.$(date -u +%H%M)
for c in 0 2; do say "book c$c: $(python3 ~/spd/feed/book.py add mol1-quality-c$c $c $S --note "mol1-quality BH grade fan-out (extension)")"; done
setsid nohup bash code/qb1_fanout.sh > fanout.log 2>&1 < /dev/null &
say "fanout relaunched pid $!"; rm -f EXTEND

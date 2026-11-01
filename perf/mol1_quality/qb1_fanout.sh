#!/bin/bash
# qb1 share of the TT-BH quality grade: same engine tree (main 90cef828f, ttnn 0.68.0), same frozen paired inputs
# (qb2's whxfer pack), cases walked from the end. Each chip starts when book.py hands its card to mol1-quality-c<N>.
# Log ~/mol1-quality/fanout.log; FANOUT-EXIT at the end.
set -uo pipefail
cd ~/mol1-quality; say() { echo "$(date -u +%FT%TZ) $*"; }
SHA=90cef828f; SEEDS="42 66 101 2024 8888"
if [ ! -f main/.ok ]; then
  rm -rf main && mkdir main && git -C ~/tt-bio-dev archive $SHA | tar -x -C main && echo $SHA > main/.ok || { say "archive failed"; exit 1; }
fi
if [ ! -f inputs/.ok ]; then
  mkdir -p xfer && cd xfer
  scp -q ttuser@tt-quietbox2:mol1-quality/whxfer/SUMS .
  for p in $(awk '{print $2}' SUMS); do
    ( for i in 1 2 3 4 5; do sha256sum -c --status <(grep " $p\$" SUMS) 2>/dev/null && break
        timeout 7200 rsync -q --partial ttuser@tt-quietbox2:mol1-quality/whxfer/$p . ; done ) &
  done; wait
  sha256sum -c SUMS || { say "pull sha mismatch"; exit 1; }
  mkdir -p ../inputs && cat p.* | gzip -dc | tar xf - -C ../inputs && cd ../inputs && python3 -c "
import json, hashlib, os, sys; i = json.load(open('INDEX.json'))
bad = [f for f, h in i['sha256'].items() if os.path.exists(f) and hashlib.sha256(open(f, 'rb').read()).hexdigest() != h]
print('sha mismatches', len(bad)); sys.exit(1 if bad else 0)" && touch .ok && cd .. || { say "inputs verify failed"; exit 1; }
fi
say "inputs + tree ready"
for c in 0 2; do
  ( o=mol1-quality-c$c
    until [ "$(python3 ~/spd/feed/book.py active $c 2>/dev/null)" = "$o" ]; do
      python3 ~/spd/feed/book.py next $c --except $o >/dev/null 2>&1; [ -z "$(python3 ~/spd/feed/book.py list | grep " $o ")" ] && { say "chip $c: booking gone"; exit 0; }
      sleep 60; done
    say "chip $c: window open"
    PATH=$HOME/tt-bio-dev/env/bin:$PATH PYTHONPATH=$HOME/mol1-quality/main LOCK_HOST=qb1 \
      bash code/run_tt.sh --reverse --book $o inputs tt-bh "$SEEDS" $c ) &
done
wait; say "FANOUT-EXIT"

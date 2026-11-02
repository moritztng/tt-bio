#!/bin/bash
# One host's share of a TT-BH quality arm (qb1 or qb2): engine tree SHA, the frozen paired inputs, every case, seeds
# 42+66 on all cases before 101/2024/8888 (D-orch-36's split). Each chip folds only inside a book.py window owned by
# mol1-quality-c<N> and books its next window itself when one ends, so it takes the gaps the other rows leave.
# Folds into ~/mol1-quality/OUT; on qb1 sync_qb1.sh OUT (run on qb2) moves them back. qb1 pulls qb2's inputs pack once.
#   bash bh_arm.sh [--reverse] SHA OUT CHIP [CHIP ...]   log on stdout, ARM-EXIT at the end
set -uo pipefail
REV=; [ "${1:-}" = --reverse ] && { REV=--reverse; shift; }
SHA=${1:?SHA}; OUT=${2:?OUT}; shift 2; [ $# -gt 0 ] || { echo "no chip"; exit 2; }
cd ~/mol1-quality; say() { echo "$(date -u +%FT%TZ) $*"; }
host=$(hostname); case $host in tt-quietbox2) LOCK_HOST=qb2 ;; tt-quietbox) LOCK_HOST=qb1 ;; *) say "unknown host $host"; exit 2 ;; esac
P=$(dirname "$(realpath "$0")"); REPO=tt-bio.git; [ -d $REPO ] || REPO=~/tt-bio-dev
if [ ! -f main-$SHA/.ok ]; then
  rm -rf main-$SHA && mkdir main-$SHA && git -C $REPO archive $SHA tt_bio | tar -x -C main-$SHA && echo $SHA > main-$SHA/.ok ||
    { say "archive $SHA failed"; exit 1; }
fi
if [ ! -f inputs/INDEX.json ]; then
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
# a staged case missing one of its files folds anyway on a fallback (online MSA, local template search), so it fails here
gone = [f for f in i['sha256'] if '/' in f and os.path.exists(f.split('/')[0] + '.json') and not os.path.exists(f)]
print('sha mismatches', len(bad), 'missing', len(gone), *gone); sys.exit(1 if bad or gone else 0)" && cd .. ||
    { say "inputs verify failed"; exit 1; }
fi
# the engine env; --use_template needs kalign (qb2: micromamba env; a host without it fails 9fm7's folds visibly)
export PATH=$HOME/tt-bio-dev/env/bin:$HOME/mol1-quality/env/kalign/bin:$PATH PYTHONPATH=$PWD/main-$SHA LOCK_HOST
command -v tt-bio > /dev/null || { say "no tt-bio on PATH"; exit 1; }
say "tree $SHA ready, out $OUT, chips $*"
for c in "$@"; do
  ( o=mol1-quality-c$c
    for seeds in "42 66" "101 2024 8888"; do
      while :; do
        python3 ~/spd/feed/book.py active $c 2>/dev/null | grep -qx $o ||
          { python3 ~/spd/feed/book.py add $o $c 14400 --note "mol1-quality $OUT" > /dev/null; say "chip $c: booked $(python3 ~/spd/feed/book.py list | grep " $o ")"; }
        until [ "$(python3 ~/spd/feed/book.py active $c 2>/dev/null)" = "$o" ]; do sleep 60; done
        say "chip $c: window open, seeds $seeds"
        bash "$P/run_tt.sh" $REV --book $o inputs "$OUT" "$seeds" $c > "$OUT.c$c.last" 2>&1; cat "$OUT.c$c.last"
        grep -q "TT-STOP\|card lock held" "$OUT.c$c.last" || break  # ran to the end of the case list inside the window
        sleep 60
      done
    done
    python3 ~/spd/feed/book.py rm $o > /dev/null; say "chip $c: done" ) &
done
wait; say "ARM-EXIT"

#!/bin/bash
# BH accuracy folds on one qb2 card (Blackhole), detached: books CARD through the QuietBox arbiter (book.py), holds
# its flock, then runs run_tt.sh from a `git archive` of the MODES.md pin (c52ab1702) through tt-bio-dev's env,
# so tt_bio resolves to the pinned tree, not the shared checkout. Accuracy does not need a quiet box, so one card.
# Order: opendde exact+fast, then Mol-1 exact (MODES.md: Mol-1 TT fast = exact). Resumable: run_tt.sh skips rc=0.
#   G=~/pvo-accuracy-bh CARD=3 nohup setsid bash bh_acc.sh &   (G holds inputs/ msa/ from pc, rsync -L)
set -u
G=${G:?}; CARD=${CARD:-3}; PIN=${PIN:-c52ab1702}; OWN=pvo-accuracy-bh; SPAN=${SPAN:-21600}
PY=/home/ttuser/tt-bio-dev/env/bin; T0=$G/tree-$PIN; HERE=$(cd "$(dirname "$0")" && pwd)
say() { echo "$(date -u +%FT%TZ) $*" >> $G/bh.log; }
say "start pid $$ card $CARD"
[ -d $T0 ] || { mkdir -p $T0 && git -C /home/ttuser/tt-bio-dev archive $PIN | tar -x -C $T0; } || { say "tree failed"; exit 4; }
cd /home/ttuser/spd/feed
while :; do
  python3 book.py list | grep -q " $OWN " ||
    python3 book.py add $OWN $CARD $SPAN --note "pvo-accuracy BH OpenDDE+Mol-1 accuracy folds" >> $G/book.log
  if python3 book.py active $CARD | tr " " "\n" | grep -qx $OWN; then
    exec 5>/home/ttuser/spd_qb2_card$CARD.lock; flock -w 300 5 && break
    exec 5>&-; say "card lock busy, rebook"; python3 book.py rm $OWN
  fi
  sleep 60; done
say "booking active, lock held, tree $PIN"
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:pvo-accuracy PYTHONPATH=$T0 PATH=$PY:$PATH
DATA=$G MODEL=opendde SEEDS="101 102 103" MODES="exact fast" bash $HERE/run_tt.sh >> $G/bh.log 2>&1
DATA=$G MODEL=mol1 SEEDS="101 102 103" MODES="exact" bash $HERE/run_tt.sh >> $G/bh.log 2>&1
cd /home/ttuser/spd/feed && python3 book.py rm $OWN
say done; echo ok > $G/DONE

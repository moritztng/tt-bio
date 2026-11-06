#!/bin/bash
# BH accuracy folds on one qb2 card (Blackhole), detached: books CARD through the QuietBox arbiter (book.py), holds
# its flock, then runs run_tt.sh from a `git archive` of the MODES.md pin (c52ab1702) through tt-bio-dev's env,
# so tt_bio resolves to the pinned tree, not the shared checkout. Accuracy does not need a quiet box, so one card.
# A fold starts only if it ends inside the booking (FOLD_S, worst case); otherwise the card goes back at the window
# edge and the driver books the next window and resumes (run_tt.sh skips rc=0). One driver per model and card:
#   G=~/pvo-accuracy-bh CARD=3 MODEL=opendde MODES="exact fast" nohup setsid bash bh_acc.sh &
#   G=~/pvo-accuracy-bh CARD=1 MODEL=mol1 MODES=exact FOLD_S=1500 nohup setsid bash bh_acc.sh &
# (G holds inputs/ msa/ from pc, rsync -L). MODES.md: Mol-1 TT fast = exact, so mol1 runs exact only.
set -u
G=${G:?}; CARD=${CARD:-3}; MODEL=${MODEL:?}; MODES=${MODES:?}; PIN=${PIN:-c52ab1702}; SPAN=${SPAN:-10800}
FOLD_S=${FOLD_S:-900}; OWN=${OWN:-pvo-accuracy-bh-$MODEL}; PY=/home/ttuser/tt-bio-dev/env/bin; T0=$G/tree-$PIN
HERE=$(cd "$(dirname "$0")" && pwd); FEED=/home/ttuser/spd/feed
say() { echo "$(date -u +%FT%TZ) $*" >> $G/bh-$MODEL.log; }
say "start pid $$ card $CARD $MODEL $MODES"
[ -d $T0 ] || { mkdir -p $T0 && git -C /home/ttuser/tt-bio-dev archive $PIN | tar -x -C $T0; } || { say "tree failed"; exit 4; }
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:pvo-accuracy PYTHONPATH=$T0 PATH=$PY:$PATH
while :; do
  cd $FEED
  while :; do
    python3 book.py list | grep -q " $OWN " ||
      python3 book.py add $OWN $CARD $SPAN --note "pvo-accuracy BH $MODEL accuracy folds" >> $G/book.log
    if python3 book.py active $CARD | tr " " "\n" | grep -qx $OWN; then
      exec 5>/home/ttuser/spd_qb2_card$CARD.lock; flock -w 300 5 && break
      exec 5>&-; say "card lock busy, rebook"; python3 book.py rm $OWN
    fi
    sleep 60; done
  END=$(awk -v o=$OWN '$3==o {print $2}' windows.tsv | tail -1)
  say "booking active until $(date -u -d @$END +%TZ), lock held, tree $PIN"
  export GUARD="cd $FEED && python3 book.py active $CARD | tr ' ' '\n' | grep -qx $OWN && [ \$((END - \$(date +%s))) -gt $FOLD_S ]" END
  DATA=$G MODEL=$MODEL SEEDS="101 102 103" MODES="$MODES" bash $HERE/run_tt.sh >> $G/bh-$MODEL.log 2>&1; rc=$?
  exec 5>&-; cd $FEED && python3 book.py rm $OWN
  [ $rc = 3 ] || break
  say "window spent, rebooking"; done
say "done rc=$rc"; [ $rc = 0 ] && echo ok > $G/DONE-$MODEL

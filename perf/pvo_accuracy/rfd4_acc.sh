#!/bin/bash
# RFD4 accuracy refolds on one qb2 card, detached. Same shape as bh_acc.sh: book CARD through the QuietBox
# arbiter, hold its flock, refold through the OF3 oracle from the pinned tree, hand the card back at the window
# edge and rebook until the side is done. refold.py skips a design whose confidences JSON exists, so a window
# that ends mid-side costs nothing, and --shards/--shard lets a second card take the same root.
#   G=~/pvo-accuracy-rfd4 CARD=3 SIDE=gpu-exact nohup setsid bash rfd4_acc.sh &
# G holds designs/<side>/ (pulled from pc) and gets out/<side>/evaluation_samples/ back. One oracle config for
# every side (5 recycles, 200 steps, of3_ppi) -- that is what makes GPU and TT comparable at all.
set -u
G=${G:?}; CARD=${CARD:-3}; SIDE=${SIDE:?}; PIN=${PIN:-c52ab1702}; SPAN=${SPAN:-10800}
FOLD_S=${FOLD_S:-600}; SHARDS=${SHARDS:-1}; SHARD=${SHARD:-0}
# The binder chain, not guessed: the geom.* and *_coi columns are about this chain, and grading the target
# chain instead would be wrong rather than noisy. Read it off a design before you launch.
CHAIN=${CHAIN:?set CHAIN to the designed chain id in designs/$SIDE/*_generated.cif}
OWN=${OWN:-pvo-accuracy-rfd4-$SIDE-$CARD}; PY=/home/ttuser/tt-bio-dev/env/bin; T0=$G/tree-$PIN; FEED=/home/ttuser/spd/feed
# Both QuietBoxes run the same arbiter but name their per-card lock after the box, so the lock path is an
# argument: qb1 refolds while qb2's four cards are booked out by other rows.
BOX=${BOX:-qb2}; LOCK=${LOCK:-/home/ttuser/spd_${BOX}_card$CARD.lock}
LOG=$G/rfd4-$SIDE-$CARD.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
say "start pid $$ card $CARD side $SIDE shard $SHARD/$SHARDS"
[ -d $T0 ] || { mkdir -p $T0 && git -C /home/ttuser/tt-bio-dev archive $PIN | tar -x -C $T0; } || { say "tree failed"; exit 4; }
# Model code is the pin; the refold harness is this row's and is overlaid from $G/harness/ (copied from
# wk/pvo-accuracy). Both sides get the identical overlay, so the oracle stays one configuration.
[ -d $G/harness ] && cp $G/harness/*.py $T0/perf/rfd4/
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:pvo-accuracy PYTHONPATH=$T0 PATH=$PY:$PATH
while :; do
  cd $FEED
  while :; do
    # Re-add every poll, not once: `add` replaces this owner's booking and never moves it later, so on a
    # box booked solid the start creeps earlier as the work ahead finishes. Booking once and waiting would
    # sit on the first slot the arbiter offered (11-04: 14 h out) even after a card came free in an hour.
    python3 book.py add $OWN $CARD $SPAN --note "pvo-accuracy RFD4 $SIDE refolds" >> $G/book.log
    if python3 book.py active $CARD | tr " " "\n" | grep -qx $OWN; then
      exec 5>$LOCK; flock -w 300 5 && break
      exec 5>&-; say "card lock busy, rebook"; python3 book.py rm $OWN
    fi
    sleep 60; done
  END=$(awk -v o=$OWN '$3==o {print $2}' windows.tsv | tail -1)
  say "booking active until $(date -u -d @$END +%TZ), lock held, tree $PIN"
  python3 $T0/perf/rfd4/refold.py --inputs $G/designs/$SIDE --out $G/out/$SIDE --family binder \
    --chain $CHAIN --shards $SHARDS --shard $SHARD --deadline $END --fold-s $FOLD_S \
    --record $G/out/$SIDE/refolds-$SHARD.jsonl >> $LOG 2>&1; rc=$?
  exec 5>&-; cd $FEED && python3 book.py rm $OWN
  [ $rc = 3 ] || break
  say "window spent, rebooking"; done
say "done rc=$rc"; [ $rc = 0 ] && echo ok > $G/DONE-$SIDE-$SHARD

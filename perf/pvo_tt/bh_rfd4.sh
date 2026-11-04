#!/bin/bash
# pvo-tt: the 128 RFD4 designs pvo-accuracy grades (state/pvo/RFD4-ACC.md), on Blackhole.
# The WH shards cannot run -- .111 went back to JapanFold and no Galaxy can be lent (spd-galaxy 08:5xZ) -- so the
# orchestrator put this arm on qb1's four cards behind bh.sh's timing cells (note-pvo-orchestrator-to-pvo-tt-rfd4-designs-bh).
# Timing wins the window: this waits for bh.sh to exit, then takes whatever cards are free, 32 seeds a block.
#   W0=<pvo-tt worktree> G=<out dir> D=<data dir with rfd4/> WAIT_PID=<bh.sh pid> bash perf/pvo_tt/bh_rfd4.sh
# Seeds 1234..1361 are the GPU side's own block; a seed whose CIF is on disk is skipped, so a lost card resumes.
set -u
W=${W0:?}; G=${G:?}; D=${D:?}; PIN=${PIN:-c52ab1702}; OWN=pvo-tt-bh-rfd4; SPAN=${SPAN:-5400}
PY=/home/ttuser/tt-bio-dev/env/bin/python; T0=$G/tree; N=${N:-32}; S0=${S0:-1234}
mkdir -p $G; say() { echo "$(date -u +%FT%TZ) $*" >> $G/rfd4.log; }
say "start pid $$ waiting on ${WAIT_PID:-none}"
# 1. timing first: do not even book while bh.sh holds the cards.
while [ -n "${WAIT_PID:-}" ] && kill -0 ${WAIT_PID} 2>/dev/null; do sleep 60; done
say "timing arm gone, booking"
cd /home/ttuser/spd/feed
end() { awk -v o=$OWN '$3 == o {print $2}' /home/ttuser/spd/feed/windows.tsv; }
# 2. same booking discipline as bh.sh: arbiter window first, then the four card flocks, else give the booking back.
tries=0
while :; do
  tries=$((tries + 1)); [ $tries -gt 240 ] && { say "no window in 4 h, giving up"; echo nowindow > $G/DONE; exit 5; }
  python3 book.py list | grep -q " $OWN " ||
    python3 book.py add $OWN 0,1,2,3 $SPAN --note "pvo-tt BH RFD4 designs for pvo-accuracy (4 cards, 32 seeds each)" >> $G/book.log
  if python3 book.py active 0 | tr " " "\n" | grep -qx $OWN; then
    # Designs are graded for QUALITY, not timed, so this arm needs no quiet box and no particular card count:
    # take whatever is free right now. Demanding all four made it as unwinnable as the timing arm for no gain.
    H=()
    for i in 0 1 2 3; do eval "exec $((5+i))>/home/ttuser/spd_qb1_card$i.lock"; flock -n $((5+i)) && H+=($i) || eval "exec $((5+i))>&-"; done
    [ ${#H[@]} -ge 1 ] && break
    say "no card free, holding the booking and retrying"
  fi
  sleep 60; done
say "booking active, locks held, window ends $(end)"
[ -d $T0 ] || git -C $W worktree add -q --detach $T0 $PIN || { say "tree failed"; echo tree > $G/DONE; exit 4; }
say "tree $(git -C $T0 rev-parse HEAD)"
# The harness lives at the branch head, not in the pin, so it is copied in beside the tree exactly as the WH shards run it.
cp $W/perf/pvo_tt/rfd4_designs.py $G/rfd4_designs.py
say "harness md5 $(md5sum $G/rfd4_designs.py | cut -c1-12)"
# $1=card $2=out dir $3=n designs $4=first seed
shard() { TT_VISIBLE_DEVICES=$1 TT_BIO_LEASE_CARDS=$1 TT_BIO_LEASE_HOLDER=worker:pvo-tt PYTHONPATH=$T0:$T0/perf:$G \
  HF_HOME=$HOME/.cache/huggingface timeout ${T:-5100} $PY $G/rfd4_designs.py \
  --spec $D/rfd4/9TH6-A-b120.yaml --weights nvidia/RFD4-Proteina --out $2 --n $3 --seed0 $4 --batch 1; }
# 3. 568 tokens has never been run on Blackhole (tt_bio/size_limits.py records no RFD4 BH ceiling), so one design on
#    one card decides whether to spend the other three.
cd $D/rfd4
shard ${H[0]} $G/s0 1 $S0 > $G/probe.log 2>&1; r=$?; echo $r > $G/probe.rc; say "probe rc=$r on card ${H[0]}"
[ $r = 0 ] || { say "568 tok does not run on p150a, no fan-out"; echo probe-failed > $G/DONE; cd /home/ttuser/spd/feed && python3 book.py rm $OWN; exit 0; }
# 4. four cards, 32 consecutive seeds each; card 0 skips the probe's seed because its CIF is already on disk.
# The 4 seed blocks are dealt round-robin over the cards we actually hold, so all 128 seeds are covered
# whether that is one card or four; a card simply runs its blocks one after another.
for j in "${!H[@]}"; do (
  for k in 0 1 2 3; do [ $((k % ${#H[@]})) = $j ] || continue
    shard ${H[$j]} $G/s$k $N $((S0 + k * N)) > $G/s$k.log 2>&1; echo $? > $G/s$k.rc
    say "block $k on card ${H[$j]} rc=$(cat $G/s$k.rc)"
  done ) & done
wait
say "blocks rc: $(for k in 0 1 2 3; do printf '%s=%s ' $k "$(cat $G/s$k.rc 2>/dev/null)"; done)"
say "designs on disk: $(ls $G/s*/designs/*_generated.cif 2>/dev/null | wc -l)"
cd /home/ttuser/spd/feed && python3 book.py rm $OWN
git -C $W worktree remove --force $T0
say done; echo ok > $G/DONE

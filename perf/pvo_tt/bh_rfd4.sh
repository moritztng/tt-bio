#!/bin/bash
# pvo-tt on Blackhole: first the RFD4 reconcile cells, then the 128 designs pvo-accuracy grades.
# The WH shards cannot run -- .111 went back to JapanFold and no Galaxy can be lent (spd-galaxy 08:5xZ) -- so the
# orchestrator put the design arm on qb1 (note-pvo-orchestrator-to-pvo-tt-rfd4-designs-bh).
#   W0=<pvo-tt worktree> G=<out dir> D=<data dir with rfd4/> [WAIT_PID=<pid>] bash perf/pvo_tt/bh_rfd4.sh
#
# The reconcile answers the orchestrator's note of 13:4xZ (the 168 s WH cell against rfd4/WH.md's ladder). It runs
# pdl1-256, pdl1-512 and the PVO 9TH6 target on ONE card, one tree, one harness, so the size slope is measured
# instead of extrapolated. rfd4/WH.md's ladder grows the BINDER and holds the target at 115 residues; the PVO cell
# has a 448-residue target, so the two differ in conditioned structure, not only in token count. The 620 cell is
# also the 568-token Blackhole fit probe, which gates the fan-out.
# Seeds 1234..1361 are the GPU side's block for the graded designs; the reconcile uses 9xxx so it never joins that
# set. A seed whose CIF is on disk is skipped, so a lost card resumes.
set -u
W=${W0:?}; G=${G:?}; D=${D:?}; PIN=${PIN:-c52ab1702}; OWN=pvo-tt-bh-rfd4; SPAN=${SPAN:-5400}
PY=/home/ttuser/tt-bio-dev/env/bin/python; T0=$G/tree; N=${N:-32}; S0=${S0:-1234}
mkdir -p $G; say() { echo "$(date -u +%FT%TZ) $*" >> $G/rfd4.log; }
say "start pid $$ waiting on ${WAIT_PID:-none}"
while [ -n "${WAIT_PID:-}" ] && kill -0 ${WAIT_PID} 2>/dev/null; do sleep 60; done
cd /home/ttuser/spd/feed
end() { awk -v o=$OWN '$3 == o {print $2}' /home/ttuser/spd/feed/windows.tsv; }
# CARDS: the cards this arm may touch. With NOBOOK=1 they are already booked to this row by the fleet
# (the dispatcher books `fleet-<slug>` for the row's own card grant), so booking them again under a second
# owner would only queue us behind ourselves.
CARDS=${CARDS:-0,1,2,3}; CL=$(echo $CARDS | tr , " ")
tries=0
while :; do
  tries=$((tries + 1)); [ $tries -gt 240 ] && { say "no window in 4 h, giving up"; echo nowindow > $G/DONE; exit 5; }
  if [ "${NOBOOK:-0}" = 1 ]; then mine=1; else
    python3 book.py list | grep -q " $OWN " ||
      python3 book.py add $OWN $CARDS $SPAN --note "pvo-tt BH RFD4 reconcile + designs" >> $G/book.log
    python3 book.py active ${CL%% *} | tr " " "\n" | grep -qx $OWN && mine=1 || mine=0
  fi
  if [ $mine = 1 ]; then
    # Designs are graded for QUALITY, not timed, so this arm needs no quiet box and no particular card count:
    # take whatever is free right now. The reconcile is a RATIO between cells on one card, so a busy box moves
    # both cells together; its clock is recorded per design and the absolute seconds are quoted only if it is sane.
    H=()
    for i in $CL; do eval "exec $((5+i))>/home/ttuser/spd_qb1_card$i.lock"; flock -n $((5+i)) && H+=($i) || eval "exec $((5+i))>&-"; done
    [ ${#H[@]} -ge 1 ] && break
    say "no card free, holding the booking and retrying"
  fi
  sleep 60; done
say "booking active, locks held on cards ${H[*]}, window ends $(end)"
[ -d $T0 ] || git -C $W worktree add -q --detach $T0 $PIN || { say "tree failed"; echo tree > $G/DONE; exit 4; }
say "tree $(git -C $T0 rev-parse HEAD)"
# The harness lives at the branch head, not in the pin, so it is copied in beside the tree exactly as the WH shards run it.
cp $W/perf/pvo_tt/rfd4_designs.py $G/rfd4_designs.py
say "harness md5 $(md5sum $G/rfd4_designs.py | cut -c1-12)"
# $1=card $2=out dir $3=n designs $4=first seed $5=spec
cell() { TT_VISIBLE_DEVICES=$1 TT_BIO_LEASE_CARDS=$1 TT_BIO_LEASE_HOLDER=worker:pvo-tt PYTHONPATH=$T0:$T0/perf:$G \
  HF_HOME=$HOME/.cache/huggingface timeout ${T:-5100} $PY $G/rfd4_designs.py \
  --spec $5 --weights nvidia/RFD4-Proteina --out $2 --n $3 --seed0 $4 --batch 1; }
cd $D/rfd4
# 3. RECONCILE, on one card, smallest first. Each cell's first design pays kernel compilation (36-77 s on WH) and is
#    dropped when the slope is read; n=4 leaves 3 warm designs per cell. The 620 cell gates everything after it.
C0=${H[0]}
for r in 256:9001:4:$D/rfd4/pdl1-256.yaml 512:9101:4:$D/rfd4/pdl1-512.yaml 620:9201:4:$D/rfd4/9TH6-A-b120.yaml; do
  nm=${r%%:*}; rest=${r#*:}; sd=${rest%%:*}; rest=${rest#*:}; nn=${rest%%:*}; sp=${rest#*:}
  T=1800 cell $C0 $G/recon/$nm $nn $sd $sp > $G/recon-$nm.log 2>&1; rc=$?; echo $rc > $G/recon-$nm.rc
  say "reconcile $nm rc=$rc $(grep -o '\"wall_s\": [0-9.]*' $G/recon/$nm/designs.jsonl 2>/dev/null | tail -3 | tr '\n' ' ')"
done
r=$(cat $G/recon-620.rc)
[ $r = 0 ] || { say "568 tok (size 620) does not run on p150a, no fan-out"; echo probe-failed > $G/DONE; [ "${NOBOOK:-0}" = 1 ] || { cd /home/ttuser/spd/feed && python3 book.py rm $OWN; }; exit 0; }
# 4. 32 consecutive seeds a block, dealt round-robin over the cards actually held, so all 128 seeds are covered
#    whether that is one card or four; a card runs its blocks one after another.
for j in "${!H[@]}"; do (
  for k in 0 1 2 3; do [ $((k % ${#H[@]})) = $j ] || continue
    cell ${H[$j]} $G/s$k $N $((S0 + k * N)) $D/rfd4/9TH6-A-b120.yaml > $G/s$k.log 2>&1; echo $? > $G/s$k.rc
    say "block $k on card ${H[$j]} rc=$(cat $G/s$k.rc)"
  done ) & done
wait
say "blocks rc: $(for k in 0 1 2 3; do printf '%s=%s ' $k "$(cat $G/s$k.rc 2>/dev/null)"; done)"
say "designs on disk: $(ls $G/s*/designs/*_generated.cif 2>/dev/null | wc -l)"
[ "${NOBOOK:-0}" = 1 ] || { cd /home/ttuser/spd/feed && python3 book.py rm $OWN; }
git -C $W worktree remove --force $T0
say done; echo ok > $G/DONE

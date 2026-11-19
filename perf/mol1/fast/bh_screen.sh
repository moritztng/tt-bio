#!/bin/bash
# mol1-fast BH screen: arms x cells on ONE qb1 p150a card, booked through the card arbiter (~/spd/feed/book.py) and held
# by its card flock, from a detached tree at PIN. A screen, not the priced number: run.py --part dev (no quiet-box
# gate), AICLK still sampled in-round; the priced cells rerun later in a quiet four-card window.
#   G=<out dir> PIN=<sha> CARD=<n> ARMS="n f0 b8 fn fb" CELLS="9DBP 9W3L" bash perf/mol1/fast/bh_screen.sh
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); PY=/home/ttuser/tt-bio-dev/env/bin/python; D=${D:-/home/ttuser/pvo-tt/data}
G=${G:?}; PIN=${PIN:?}; C=${CARD:-0}; OWN=mol1-fast-bh; T0=$G/tree; SPAN=${SPAN:-14400}
mkdir -p $G; say() { echo "$(date -u +%FT%TZ) $*" >> $G/bh.log; }
say "start pid $$ card $C pin $PIN"
cd /home/ttuser/spd/feed
while :; do
  python3 book.py list | grep -q " $OWN " || python3 book.py add $OWN $C $SPAN --note "mol1-fast BH lever screen (one card)" >> $G/book.log
  if python3 book.py active $C | tr " " "\n" | grep -qx $OWN; then
    exec 5>/home/ttuser/spd_qb1_card$C.lock; flock -w 60 5 && break; exec 5>&-; say "card lock busy, retrying"
  fi
  sleep 60; done
say "booking active, lock held"
[ -d $T0 ] || git -C $W worktree add -q --detach $T0 $PIN || { say "tree failed"; echo tree > $G/DONE; exit 4; }
. $T0/perf/mol1/fast/arms.sh
FW=$(~/.local/bin/tt-smi -s 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)[\"device_info\"][0][\"firmwares\"][\"fw_bundle_version\"])")
say "tree $(git -C $T0 rev-parse HEAD) fw $FW"
cd $D/inputs
for c in ${CELLS:-9DBP 9W3L}; do for a in ${ARMS:-n f0 b8 fn fb}; do
  o=$G/$a-$c; grep -qsx 0 $o.rc && continue; arm_env bh $a || continue
  TT_BIO_LEVERS=$LV TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:mol1-fast PYTHONPATH=$T0 timeout 5400 \
    $PY $T0/perf/mol1/fair/run.py --chips $C --part dev --firmware "$FW" --model mol1 --arm $a --rounds 3 --max-attempts 4 \
    --input $D/inputs/$c.yaml --out $o --cli-extra --diffusion_samples 5 --recycling_steps 20 --msa_dir $D/msa/cache --msa_cache_only $FASTFLAG > $o.log 2>&1
  echo $? > $o.rc; say "$a $c rc=$(cat $o.rc)"
done; done
cd /home/ttuser/spd/feed && python3 book.py rm $OWN
git -C $W worktree remove --force $T0
say done; echo ok > $G/DONE

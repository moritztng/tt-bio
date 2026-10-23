#!/usr/bin/env bash
# rfd4-wh-bringup, qb2: the Blackhole side of the Wormhole accuracy grade and the paired BH no-regression check.
# Waits for the orchestrator's booking of card 2 (book.py, owner rfd4-wh-bringup) and for no holder on the card, then
# runs on ONE chip under its flock, same seeds and specs as the WH arms on .103 (pdl1, --rng upstream, seeds 0-30):
#   br256 / br512  this branch's tree: the BH arm the WH designs are graded against, and the "after" side
#   mn256 / mn512  main 01c1644ae: the "before" side; BH must come out byte-identical (the branch only gates a
#                  Wormhole-only form), and its timings pair with br on the same chip
#   z8256 / z8512  the tip tree carrying pair_bfp8z ($R/z8), the BH side of its Wormhole grade (D-orch-136)
#   [CARD=1 LEGS="z8256 z8512"] nohup setsid bash bh_chain.sh > /dev/null 2>&1 &
set -u
R=$HOME/scratch/rfd4-wh-bringup; BOOK="python3 $HOME/spd/feed/book.py"; CARD=${CARD:-2}
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python; W=$HOME/rfd4-bundle; IN=$R/inputs
exec >>"$R/chain.log" 2>&1
echo "chain start $(date -u +%FT%TZ) pid=$$"
held(){ for p in /proc/[0-9]*; do for f in $p/fd/*; do [ "$(readlink "$f" 2>/dev/null)" = /dev/tenstorrent/$CARD ] &&
          echo "${p#/proc/}:$(cat $p/comm 2>/dev/null)"; done; done 2>/dev/null | sort -u; }
deadline=$(( $(date +%s) + 6*3600 ))
while :; do
  read -r START END < <($BOOK add rfd4-wh-bringup $CARD 5400 --note "RFD4 paired BH 31-seed arm for QUALITY-WH.json" 2>/dev/null | tail -1)
  now=$(date +%s); h=$(held | tr "\n" " ")
  echo "$(date -u +%FT%TZ) booking ${START:-?}-${END:-?} holders: ${h:-none}"
  if [ -n "${START:-}" ] && [ "$now" -ge "$START" ] && [ -z "$h" ]; then break; fi
  [ "$now" -gt "$deadline" ] && { echo "GIVE-UP: no window in 6 h"; exit 4; }
  sleep 60
done
echo "window open $(date -u +%FT%TZ)"; uptime
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:rfd4-wh-bringup
for leg in ${LEGS:-br256 mn256 br512 mn512}; do
  case ${leg:0:2} in mn) tree=$R/main;; z8) tree=$R/z8;; *) tree=$R/branch;; esac
  n=${leg:2}; seeds="$(seq 0 30)"; [ "$leg" = mn512 ] && seeds="$(seq 0 7)"
  echo "$leg start $(date -u +%FT%TZ) tree $(cat $tree/SHA) load $(cut -d' ' -f1 /proc/loadavg)"
  (cd $tree && PYTHONPATH=$tree flock -w 900 $HOME/spd_qb2_chip$CARD.lock timeout 3000 $PY perf/rfd4/e2e/design.py \
     --spec $IN/pdl1-$n.yaml --weights $W --rng upstream --seeds $seeds --out $R/runs/$leg) > $R/runs/$leg.log 2>&1
  echo "$leg rc=$? $(date -u +%FT%TZ)"
done
$BOOK rm rfd4-wh-bringup
echo "chain end $(date -u +%FT%TZ)"; touch $R/CHAIN-DONE

#!/usr/bin/env bash
# rfd4-bh-scaling chain: book qb2 chips 0,1,3, wait for the window and for the chips to be free, then measure.
set -u
R=$HOME/scratch/rfd4-bh-scaling; WT=$HOME/.coworker/wt/rfd4-bh-scaling; BOOK="python3 $HOME/spd/feed/book.py"
CARDS=0,1,3; PY=$HOME/tt-bio-dev/env/bin/python
cd "$WT" || exit 1
exec >>"$R/chain3.log" 2>&1
echo "chain3 start $(date -u +%FT%TZ) pid=$$"
held(){ for p in /proc/[0-9]*; do for f in $p/fd/*; do case $(readlink "$f" 2>/dev/null) in /dev/tenstorrent/0|/dev/tenstorrent/1|/dev/tenstorrent/3) echo "${p#/proc/}:$(cat $p/comm 2>/dev/null):$(readlink $f)";; esac; done; done 2>/dev/null | sort -u; }
deadline=$(( $(date +%s) + 14*3600 ))
while :; do
  read -r START END < <($BOOK add rfd4-bh-scaling $CARDS 3600 --note "rfd4-bh-scaling 3 chips with dedicated cores" 2>/dev/null | tail -1)
  now=$(date +%s); h=$(held | tr "\n" " ")
  echo "$(date -u +%FT%TZ) booking ${START:-?}-${END:-?} holders: ${h:-none}"
  if [ -n "${START:-}" ] && [ "$now" -ge "$START" ] && [ -z "$h" ]; then break; fi
  [ "$now" -gt "$deadline" ] && { echo "GIVE-UP: no window in 14 h"; $BOOK rm rfd4-bh-scaling; exit 4; }
  sleep 60
done
echo "window open $(date -u +%FT%TZ)"; uptime
export PYTHONPATH="$WT" TT_BIO_LEASE_HOLDER=worker:rfd4-bh-scaling
ROUNDS="0;1;3;0,1,3;0,1,3@1;0,1,3@2;0,1,3@3"
for S in 256 224; do
  echo "size $S start $(date -u +%FT%TZ)"
  timeout 1700 $PY perf/rfd4/scaling/scale.py run --cards $CARDS --spec $R/inputs/pdl1-$S.yaml --weights $HOME/rfd4-bundle \
      --batch 4 --out $R/runs3/s$S --reps 4 --rounds "$ROUNDS" --tag s$S > $R/runs3-s$S.out 2>&1
  echo "size $S rc=$? $(date -u +%FT%TZ)"
  $PY perf/rfd4/scaling/scale.py report $R/runs3/s$S > /dev/null 2>&1 && echo "report s$S ok"
done
$BOOK rm rfd4-bh-scaling
echo "chain3 end $(date -u +%FT%TZ)"; touch $R/CHAIN2-DONE

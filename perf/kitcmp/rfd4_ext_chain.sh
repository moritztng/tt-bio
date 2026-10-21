#!/bin/bash
# Own the rest of session 3's life: pull every 3 min, run the extended batch climb once the box's own phases
# finish, pull again, destroy. Replaces rfd4_session.sh's watch loop, which destroys at RFD4-DONE and so would
# take the box away before the climb that the fair number actually needs.
#   ./rfd4_ext_chain.sh <results-dir> <instance-id> <keep-until-epoch>
# The climb is the point: session 3 screened and swept under a 16/8/4 ladder while the B200 held 8.5 GiB of
# ~180 GiB, so the ladder, not the card, set NVIDIA's rate. `--sweep-only` re-climbs each size's winning arm
# on this same box, reusing the rungs already measured.
set -u
R=${1:?results dir}; I=${2:?instance}; KEEP=${3:?keep-until epoch}
API=https://console.vast.ai/api/v0; K=$(cat ~/.config/vastai/vast_api_key)
S=$(cat "$R/ssh"); H=root@$(sed -n 's/.*root@\([0-9.]*\).*/\1/p' <<<"$S")
PULL(){ rsync -a -e "${S% root@*}" "$H:/root/kc/rfd4/" "$R/rfd4/" 2>/dev/null; }
say(){ echo "$(date -u +%FT%TZ) ext_chain: $*"; }
destroy(){ say "destroying $I: $*"; curl -s -X DELETE -H "Authorization: Bearer $K" "$API/instances/$I/" >/dev/null
           echo "$(date -u +%FT%TZ) $*" >> "$R/SESSION-END"; exit 0; }

say "watching $I until $(date -u -d @"$KEEP" +%FT%TZ)"
MISS=0
while :; do
  sleep 180
  PULL
  if ! timeout 30 $S true 2>/dev/null; then
    MISS=$((MISS+1)); say "ssh miss $MISS/5"
    [ $MISS -ge 5 ] && destroy "box unreachable 5 polls running"
    continue
  fi
  MISS=0
  [ "$(date +%s)" -ge "$KEEP" ] && destroy "keep-until reached"
  # RFD4-FAILED is terminal: the box's own gate already said no arm measured anything.
  timeout 30 $S 'ls /root/kc/rfd4/RFD4-FAILED' >/dev/null 2>&1 && { PULL; destroy "box reported RFD4-FAILED"; }
  # The box writes RFD4-DONE after its reference designs; that is when the GPU is free for the climb.
  timeout 30 $S 'ls /root/kc/rfd4/RFD4-DONE' >/dev/null 2>&1 || continue

  say "box phases done -- starting the extended batch climb"
  rsync -a -e "${S% root@*}" "$(dirname "$0")/rfd4_arms.py" "$H:/root/kc/rfd4_arms.py" || destroy "climb push failed"
  timeout 30 $S "rm -f /root/kc/rfd4/ARMS-EXT.json" >/dev/null 2>&1
  # Deadline for the climb: 15 min of headroom under keep-until, so the pull below always has time to run.
  timeout 30 $S "cd /root/kc && setsid nohup /opt/rfd4-venv/bin/python rfd4_arms.py --inputs /root/kc/inputs \
      --out /root/kc/rfd4 --deadline $((KEEP-900)) --sweep-only --screen-warm 5 --final-warm 9 \
      > /root/kc/rfd4/sweep-ext.log 2>&1 < /dev/null &" || destroy "climb launch failed"
  say "climb launched"
  while :; do
    sleep 180
    PULL
    timeout 30 $S true 2>/dev/null || destroy "box went away during the climb (partial climb pulled)"
    [ "$(date +%s)" -ge "$KEEP" ] && { PULL; destroy "keep-until reached during the climb"; }
    timeout 30 $S 'pgrep -f "rfd4_arms.py --inputs" >/dev/null' 2>/dev/null && continue
    PULL; destroy "extended climb finished"
  done
done

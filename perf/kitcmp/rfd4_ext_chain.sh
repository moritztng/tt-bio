#!/bin/bash
# Own the rest of session 3's life: pull every 3 min, run the extended batch climb once the box's own phases
# finish, then regenerate the reference designs from the arm the climb actually chose, pull, destroy.
#   ./rfd4_ext_chain.sh <results-dir> <instance-id> <keep-until-epoch>
#
# Two reasons this exists instead of rfd4_session.sh's watch loop:
#   * that loop destroys the box at RFD4-DONE, which is before the climb the fair number needs;
#   * the box generates its reference designs from the SCREEN's winning arm, and the climb can re-rank (the
#     screen ranks arms at one small batch, where the card is 22-50 % busy, so it partly ranks host overhead).
#     Reference designs must come from the arm we publish, so they are regenerated here when the arm changes.
# Every exit path destroys the box; artifacts/rfd4-fair-gpu/tick.d/box_reaper.sh covers this script dying.
set -u
R=${1:?results dir}; I=${2:?instance}; KEEP=${3:?keep-until epoch}
API=https://console.vast.ai/api/v0; K=$(cat ~/.config/vastai/vast_api_key)
S=$(cat "$R/ssh"); H=root@$(sed -n 's/.*root@\([0-9.]*\).*/\1/p' <<<"$S")
PULL(){ rsync -a -e "${S% root@*}" "$H:/root/kc/rfd4/" "$R/rfd4/" 2>/dev/null; }
say(){ echo "$(date -u +%FT%TZ) ext_chain: $*"; }
destroy(){ say "destroying $I: $*"; curl -s -X DELETE -H "Authorization: Bearer $K" "$API/instances/$I/" >/dev/null
           echo "$(date -u +%FT%TZ) $*" >> "$R/SESSION-END"; exit 0; }
alive(){ timeout 30 $S true 2>/dev/null; }
busy(){ timeout 30 $S 'pgrep -f "rfd4_arms.py --inputs" >/dev/null' 2>/dev/null; }
# Wait for a launched phase to finish, pulling as it goes. Any exit here is terminal for the session.
await(){
  local what=$1
  while :; do
    sleep 180; PULL
    alive || destroy "box went away during $what (partial results pulled)"
    [ "$(date +%s)" -ge "$KEEP" ] && { PULL; destroy "keep-until reached during $what"; }
    busy || { PULL; say "$what finished"; return 0; }
  done
}

say "watching $I until $(date -u -d @"$KEEP" +%FT%TZ)"
MISS=0
while :; do
  sleep 180
  PULL
  if ! alive; then
    MISS=$((MISS+1)); say "ssh miss $MISS/5"
    [ $MISS -ge 5 ] && destroy "box unreachable 5 polls running"
    continue
  fi
  MISS=0
  [ "$(date +%s)" -ge "$KEEP" ] && destroy "keep-until reached"
  timeout 30 $S 'ls /root/kc/rfd4/RFD4-FAILED' >/dev/null 2>&1 && { PULL; destroy "box reported RFD4-FAILED"; }
  timeout 30 $S 'ls /root/kc/rfd4/RFD4-DONE' >/dev/null 2>&1 || continue

  say "box phases done -- extended batch climb next"
  rsync -a -e "${S% root@*}" "$(dirname "$0")/rfd4_arms.py" "$H:/root/kc/rfd4_arms.py" || destroy "climb push failed"
  timeout 30 $S "rm -f /root/kc/rfd4/ARMS-EXT.json" >/dev/null 2>&1
  timeout 30 $S "cd /root/kc && setsid nohup /opt/rfd4-venv/bin/python rfd4_arms.py --inputs /root/kc/inputs \
      --out /root/kc/rfd4 --deadline $((KEEP-1800)) --sweep-only --screen-warm 5 --final-warm 9 \
      > /root/kc/rfd4/sweep-ext.log 2>&1 < /dev/null &" || destroy "climb launch failed"
  say "climb launched"
  await "the climb"

  # The arm the climb chose at 256, which is the size the reference designs are generated at.
  ARM=$(timeout 30 $S "/opt/rfd4-venv/bin/python -c \"
import json
try:
    e = json.load(open('/root/kc/rfd4/ARMS-EXT.json'))
    print(e.get('256', {}).get('arm') or '')
except Exception:
    print('')\"" 2>/dev/null | tr -d '\r\n ')
  WAS=$(timeout 30 $S "/opt/rfd4-venv/bin/python -c \"
import json; print(json.load(open('/root/kc/rfd4/ARMS.json')).get('best_arm') or '')\"" 2>/dev/null | tr -d '\r\n ')
  say "climb chose '${ARM:-none}' at 256; the box's reference designs used '${WAS:-none}'"
  if [ -n "$ARM" ] && [ "$ARM" != "$WAS" ] && [ "$(date +%s)" -lt $((KEEP-1800)) ]; then
    say "arm changed -- regenerating the reference designs from $ARM"
    timeout 30 $S "cd /root/kc && setsid nohup /opt/rfd4-venv/bin/python rfd4_arms.py --inputs /root/kc/inputs \
        --out /root/kc/rfd4 --deadline $((KEEP-600)) --reference 30 --reference-arm $ARM \
        > /root/kc/rfd4/reference-ext.log 2>&1 < /dev/null &" || destroy "reference relaunch failed"
    await "the reference designs"
  fi
  PULL; destroy "climb and reference designs complete"
done

#!/bin/bash
# Pull the extension phases off an already-rented box and destroy it when they finish or at a hard cap.
#   rfd4_ext_watch.sh <results-dir> <instance-id> <hard-cap-epoch>
# rfd4_session.sh destroys the box the moment box_rfd4.sh writes RFD4-DONE, which is right where the
# extension phases need to start, so its watcher is stopped and this one takes over the two jobs that
# watcher owned: pulling /root/kc/rfd4 every 3 minutes, and destroying the instance at the end.
# A rented box must never depend on one process staying alive: state/rfd4/gpu/{OWNED,KEEP-UNTIL} stay armed
# and artifacts/rfd4-fair-gpu/tick.d/box_reaper.sh destroys the instance if this watcher dies.
set -u
OUT=${1:?results dir} I=${2:?instance id} CAP=${3:?hard cap epoch}
S=$(cat "$OUT/ssh"); H=$(echo "$S" | grep -oP 'root@\S+')
K=$(cat ~/.config/vastai/vast_api_key); API=https://console.vast.ai/api/v0
B=/home/moritz/.coworker/state/vast-budget
log(){ echo "$(date -u +%FT%TZ) ext_watch: $*"; }
pull(){ rsync -a -e "${S% root@*}" "$H:/root/kc/rfd4/" "$OUT/rfd4/" 2>/dev/null
        rsync -a -e "${S% root@*}" "$H:/root/kc/*.log" "$OUT/" 2>/dev/null; }

log "watching $I until $(date -u -d @$CAP +%FT%TZ)"
END=""
while :; do
  sleep 180
  pull
  [ -n "$(timeout 30 $S 'ls /root/kc/rfd4/EXT-DONE' 2>/dev/null)" ] && END=ext-done
  [ $(date +%s) -ge $CAP ] && END=${END:-cap}
  # A box that has gone away stops the watcher; it does not leave it polling a dead host for hours.
  timeout 30 $S true 2>/dev/null || { timeout 30 $S true 2>/dev/null || END=${END:-unreachable}; }
  [ -n "$END" ] && break
done
pull
log "$I ended: $END"
curl -s -X DELETE -H "Authorization: Bearer $K" "$API/instances/$I/" >/dev/null; sleep 10
N=$(curl -s -H "Authorization: Bearer $K" "$API/instances/?owner=me" |
      python3 -c 'import json,sys;print(len(json.load(sys.stdin)["instances"]))' 2>/dev/null)
log "DESTROYED $I ($END); instances left: ${N:-?}"
sed -i "s|^$I  .*|#$I DESTROYED $(date -u +%FT%TZ) (ext: $END), confirmed instances=${N:-?}|" $B
# The reaper only ever touches ids listed here, so an id that is gone must leave the list.
sed -i "/^$I\$/d" /home/moritz/.coworker/state/rfd4/gpu/OWNED
echo "$END" > "$OUT/EXT-END"

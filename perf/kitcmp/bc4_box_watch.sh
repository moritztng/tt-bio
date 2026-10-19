#!/bin/bash
# Pull results from a RUNNING bc4 rental and destroy it at a deadline. Standalone on purpose: it adopts a box
# whose driver is gone.
#   usage: bc4_box_watch.sh <label> <instance_id> <destroy_at_epoch>
#
# Two things killed the driver this exists for:
#  * bc4_quality_box.sh runs under whatever launched it. On 2026-10-20 the agent harness reaped the process
#    group it had been started in and took the driver with it, 3 minutes after the box started computing. The
#    box kept billing with nothing left on pc to pull from it or destroy it.
#  * A rental with no reaper does not stop at a cap, it stops at the credit floor.
# So the deadline is an ABSOLUTE epoch, not a duration: re-adopting a box must not extend its life, and this
# script is safe to start again if it is itself killed.
set -u
L=${1:?label} I=${2:?instance id} DEADLINE=${3:?destroy-at epoch}
P=$(cd "$(dirname "$0")" && pwd); OUT=$P/results/$L
B=/home/moritz/.coworker/state/vast-budget; HOLD=/home/moritz/.coworker/state/vast-holder
K=$(cat ~/.config/vastai/vast_api_key); API=https://console.vast.ai/api/v0
S=$(cat "$OUT/ssh") || { echo "no ssh line at $OUT/ssh"; exit 2; }
H=root@${S##*root@}
log(){ echo "$(date -u +%FT%TZ) $*"; }
credit(){ curl -s -H "Authorization: Bearer $K" "$API/users/current/" | python3 -c 'import json,sys;print(json.load(sys.stdin)["credit"])'; }
state(){ curl -s -H "Authorization: Bearer $K" "$API/instances/?owner=me" | python3 -c 'import json,sys
print(next((x.get("actual_status") or "" for x in json.load(sys.stdin)["instances"] if x["id"]==int(sys.argv[1])), "gone"))' "$I"; }
R(){ rsync -a -e "${S% root@*}" "$@"; }
PULL(){ R --include='*/' --include='*.csv' --include='*.jsonl' --include='*.log' --include='*.json' \
           --include='rc' --include='SESSION-RC' --exclude='*' $H:/root/kc/bc4-quality/ $OUT/bc4-quality/ 2>/dev/null
        R $H:'/root/kc/*.log' $OUT/ 2>/dev/null; }

log "watching $I for label $L, destroy at $(date -u -d @$DEADLINE +%FT%TZ), credit \$$(credit)"
END=""
while :; do
  sleep 180
  PULL
  [ -f "$OUT/bc4-quality/SESSION-RC" ] && END=done
  [ -z "$END" ] && [ "$(state)" != running ] && END=stopped
  [ "$(date +%s)" -ge "$DEADLINE" ] && END=${END:-cap}
  [ -n "$END" ] && break
done
PULL
log "$I ended: $END, $(find $OUT/bc4-quality -name runs.jsonl 2>/dev/null | wc -l) campaign dirs pulled"
curl -s -X DELETE -H "Authorization: Bearer $K" "$API/instances/$I/" >/dev/null; sleep 10
N=$(curl -s -H "Authorization: Bearer $K" "$API/instances/?owner=me" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["instances"]))')
log "DESTROYED $I (watch: $END); instances left: $N; credit \$$(credit)"
rm -f $HOLD/$I
sed -i "s|^$I  .*|#$I DESTROYED $(date -u +%FT%TZ) (watch: $END), confirmed instances=$N|" $B
echo "$END" > $OUT/SESSION-END

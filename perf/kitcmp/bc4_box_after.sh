#!/usr/bin/env bash
# Run bc4_quality_box.sh once no rental of ours is billing any more.
#
# The vast floor check inside bc4_quality_box.sh reads the credit balance at a moment in time and cannot see a
# second box about to bill against the same balance. Two drivers started together both pass it and together
# cross the floor. This serialises them on the only thing that actually matters -- an instance count of zero --
# so a follow-on session can be queued the moment its predecessor's result is known, without a human in between.
#
#   usage: bc4_box_after.sh <max_wait_min> <label> <cap_h> ['<query>'] [attempts]
# every argument after <max_wait_min> is handed to bc4_quality_box.sh unchanged.
set -u
WAIT_MIN=${1:?usage: bc4_box_after.sh <max_wait_min> <label> <cap_h> [query] [attempts]}; shift
P=$(cd "$(dirname "$0")" && pwd)
K=$(cat ~/.config/vastai/vast_api_key); API=https://console.vast.ai/api/v0
log(){ echo "$(date -u +%FT%TZ) after: $*"; }
live(){ curl -s -H "Authorization: Bearer $K" "$API/instances/?owner=me" |
  python3 -c 'import json,sys;print(len(json.load(sys.stdin)["instances"]))' 2>/dev/null || echo unknown; }

DEADLINE=$(( $(date +%s) + WAIT_MIN * 60 ))
while :; do
  N=$(live)
  [ "$N" = 0 ] && { log "no instance billing, starting $*"; break; }
  # An unanswered API is not an empty fleet. Treat it as "still busy" and keep waiting: renting on top of a box
  # we merely failed to see is the expensive mistake, waiting one more minute is the cheap one.
  [ "$(date +%s)" -ge "$DEADLINE" ] && { log "ABANDONED: $N instance(s) still live after $WAIT_MIN min"; exit 3; }
  sleep 60
done
exec bash "$P/bc4_quality_box.sh" "$@"

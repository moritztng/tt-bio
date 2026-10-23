#!/bin/bash
# Wait for the fleet's rented GPU to go away, then start one B200 session. Detached, its own log.
#   rfd4_wait_rent.sh <label> <cap_h> <min_credit_usd> [max_wait_h]
# Two rows share this campaign's vast credit and the single-B200 market is empty today, so a session that
# rents while another row's box is live either empties the credit or takes the offer that row needs. This
# waits for both conditions instead: no instance of ours is running, and the credit still covers the cap.
# `touch state/rfd4/gpu/NO-RENT` stops it without killing anything that is already measuring.
set -u
L=${1:?label} CAP_H=${2:?cap hours} MINC=${3:?min credit} MAXW=${4:-10}
P=$(cd "$(dirname "$0")" && pwd); K=$(cat ~/.config/vastai/vast_api_key)
API=https://console.vast.ai/api/v0; S4=/home/moritz/.coworker/state/rfd4
log(){ echo "$(date -u +%FT%TZ) wait_rent($L): $*"; }
ask(){ curl -s -H "Authorization: Bearer $K" "$API/$1"; }

T0=$(date +%s)
while [ $(( $(date +%s) - T0 )) -lt $(python3 -c "print(int($MAXW*3600))") ]; do
  [ -f $S4/gpu/NO-RENT ] && { log "NO-RENT present, not renting"; exit 0; }
  N=$(ask 'instances/?owner=me' | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["instances"]))')
  C=$(ask 'users/current/' | python3 -c 'import json,sys;print(json.load(sys.stdin)["credit"])')
  if [ "${N:-1}" -eq 0 ] && python3 -c "import sys;sys.exit(0 if $C >= $MINC else 1)"; then
    log "no instance running, credit \$$C >= \$$MINC -- starting the session"
    exec bash $P/rfd4_session.sh "$L" "$CAP_H" "${VAST_Q:-gpu_name=B200 reliability>=0.90 dph<=16}"
  fi
  log "holding: $N instance(s), credit \$$C"
  sleep 300
done
log "gave up after $MAXW h"

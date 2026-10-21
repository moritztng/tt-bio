#!/bin/bash
# One unattended B200 session for the RFD4 fair B200 baseline plus rfd4-quality's reference designs, run DETACHED
# on pc. Rents the first B200 offer that passes (ssh up, power limit in window), pushes the RFD4 harness and the frozen inputs, runs
# box_rfd4.sh on the box, pulls /root/kc/rfd4 every 3 minutes, and destroys the box at RFD4-DONE or at the cap.
#   usage: rfd4_session.sh <label> <cap_h> ['<vastai search query>'] [max_attempts]
# Status on stdout (redirect it); results perf/kitcmp/results/<label>/rfd4/...
# Derived from bc4_session.sh, which proved the vast mechanics on 2026-10-19 (two routes to the box, the tried-offer
# file on disk, the credit floor, destroy-on-cap). RFD4 pushes the pinned inputs instead of a quality request.
set -u
L=${1:?usage: rfd4_session.sh <label> <cap_h> [query] [attempts]} CAP_H=${2:?cap hours} NMAX=${4:-6}
# reliability>=0.97 was the only clause that rejected a live B200 on 2026-10-19 (offer 51577286, 0.995 passes but the
# market is thin enough that one filter empties it); 0.90 keeps the junk out without emptying the list.
Q=${3:-'gpu_name=B200 num_gpus=1 cuda_max_good>=13.0 inet_down>=200 reliability>=0.90 dph<=7'}
MINW=${MINW:-990} MAXW=${MAXW:-1010} SLUG=${SLUG:-rfd4-b200}
P=$(cd "$(dirname "$0")" && pwd); V=~/.vast-venv/bin/vastai; B=/home/moritz/.coworker/state/vast-budget
HOLD=/home/moritz/.coworker/state/vast-holder; S4=/home/moritz/.coworker/state/rfd4
K=$(cat ~/.config/vastai/vast_api_key); API=https://console.vast.ai/api/v0; FLOOR=${VAST_FLOOR_USD:-1}
OUT=$P/results/$L; mkdir -p "$OUT"
# Offers we have already paid for and lost live on disk, not in this process: a relaunch after a crash used to
# re-pick the offer it had just destroyed (2026-10-19, offer 50255941 twice, ~$2.8).
TRIEDF=$OUT/tried; touch "$TRIEDF"; TRIED=" $(tr '\n' ' ' < "$TRIEDF")"
log(){ echo "$(date -u +%FT%TZ) $*"; }
credit(){ curl -s -H "Authorization: Bearer $K" "$API/users/current/" | python3 -c 'import json,sys;print(json.load(sys.stdin)["credit"])'; }
gone(){ curl -s -H "Authorization: Bearer $K" "$API/instances/?owner=me" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["instances"]))'; }
state(){ curl -s -H "Authorization: Bearer $K" "$API/instances/?owner=me" | python3 -c 'import json,sys
print(next((x.get("intended_status") or "" for x in json.load(sys.stdin)["instances"] if x["id"]==int(sys.argv[1])), "gone"))' $1; }
destroy(){ curl -s -X DELETE -H "Authorization: Bearer $K" "$API/instances/$1/" >/dev/null; sleep 10
  local n; n=$(gone); log "DESTROYED $1 ($2); instances left: $n"; rm -f $HOLD/$1
  sed -i "s|^$1  $CAP_H  $SLUG$|#$1 DESTROYED $(date -u +%FT%TZ) ($2), confirmed instances=$n|" $B; }
next_offer(){ $V search offers "$Q" -o dph --raw 2>/dev/null | python3 -c 'import json,sys
t=sys.argv[1].split()
print(next((str(o["id"]) for o in json.load(sys.stdin) if str(o["id"]) not in t), ""))' "$TRIED"; }

MAXDPH=$(echo "$Q" | grep -oP 'dph<=\K[0-9.]+'); C=$(credit)
python3 -c "import sys; sys.exit(0 if $C - $CAP_H * ${MAXDPH:-7} >= $FLOOR else 1)" ||
  { log "REFUSED: credit \$$C - $CAP_H h x \$${MAXDPH:-7}/h would cross the \$$FLOOR floor"; exit 4; }
log "credit \$$C at start, cap $CAP_H h"

for _ in $(seq $NMAX); do
  O=$(next_offer); [ -z "$O" ] && { log "no untried offer matches: $Q"; break; }
  TRIED="$TRIED$O "; echo "$O" >> "$TRIEDF"
  C=$(credit); python3 -c "import sys; sys.exit(0 if $C - $CAP_H * ${MAXDPH:-7} >= $FLOOR else 1)" ||
    { log "REFUSED at offer $O: credit \$$C would cross the \$$FLOOR floor"; exit 4; }
  echo "-  $CAP_H  $SLUG  # $L creating on offer $O" >> $B
  I=$($V create instance $O --image nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 --disk ${DISK:-150} --ssh --direct --label $SLUG-$L --raw 2>&1 | grep -oP "new_contract['\"]:\s*\K[0-9]+")
  [ -z "$I" ] && { sed -i '$d' $B; log "offer $O: create failed"; continue; }
  mkdir -p $HOLD; echo $$ > $HOLD/$I; sed -i "\$s|.*|$I  $CAP_H  $SLUG|" $B
  echo "# ^ $SLUG $L offer $O created $(date -u +%FT%TZ) by perf/kitcmp/rfd4_session.sh" >> $B
  T0=$(date +%s); log "created $I on offer $O"; S=""
  while [ $(( $(date +%s) - T0 )) -lt 720 ]; do
    [ $(( $(date +%s) - T0 )) -ge 120 ] && [ "$(state $I)" != running ] && break
    # Two routes to the same box: the host's own open port, and vast's proxy. On 2026-10-19 a box that never
    # answered on the direct port cost two rentals, so try both every poll and take whichever answers.
    ROUTES=$($V show instance $I --raw 2>/dev/null | python3 -c 'import json,sys
x=json.load(sys.stdin)
if x.get("actual_status")=="running":
    p=(x.get("ports") or {}).get("22/tcp")
    ip=(x.get("public_ipaddr") or "").strip()
    if p and ip: print(p[0]["HostPort"], ip)
    if x.get("ssh_host") and x.get("ssh_port"): print(x["ssh_port"], x["ssh_host"])' 2>/dev/null)
    while read -r PORT HOST; do
      [ -z "${PORT:-}" ] && continue
      S="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -p $PORT root@$HOST"
      timeout 30 $S true >/dev/null 2>&1 && break; S=""
    done <<< "$ROUTES"
    [ -n "$S" ] && break
    sleep 30
  done
  [ -z "$S" ] && { destroy $I "no ssh: intended_status=$(state $I)"; continue; }
  W=$(timeout 30 $S 'nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits' 2>/dev/null | tail -1 | cut -d. -f1)
  log "$I up at $HOST:$PORT, power limit ${W:-?} W"
  [ "${W:-0}" -ge "$MINW" ] && [ "${W:-0}" -le "$MAXW" ] || { destroy $I "power limit ${W:-?} W outside $MINW-$MAXW"; continue; }
  echo "$S" > $OUT/ssh; echo "$I $O" > $OUT/instance
  R(){ rsync -a -e "${S% root@*}" "$@"; }; H=root@$HOST
  $S 'mkdir -p /root/kc' &&
    R $P/rfd4_gpu.py $P/rfd4_arms.py $P/box_rfd4.sh $H:/root/kc/ &&
    R $S4/inputs/ $H:/root/kc/inputs/ && echo "$I $O" | timeout 30 $S 'cat > /root/kc/instance' ||
    { destroy $I "push failed"; continue; }
  timeout 60 $S -f "cd /root/kc && setsid nohup bash box_rfd4.sh $CAP_H > box_rfd4.log 2>&1 < /dev/null &" ||
    { destroy $I "launch failed"; continue; }
  log "$I running box_rfd4.sh"
  END=""
  while :; do
    sleep 180
    R $H:/root/kc/rfd4/ $OUT/rfd4/ 2>/dev/null; R $H:'/root/kc/*.log' $OUT/ 2>/dev/null
    st=$(timeout 30 $S 'ls /root/kc/rfd4/RFD4-DONE' 2>/dev/null)
    [ -n "$st" ] && END=done
    [ -z "$st" ] && [ "$(state $I)" != running ] && END=stopped
    [ $(( $(date +%s) - T0 )) -ge $(python3 -c "print(int($CAP_H*3600-420))") ] && END=${END:-cap}
    [ -n "$END" ] && break
  done
  R $H:/root/kc/rfd4/ $OUT/rfd4/ 2>/dev/null; R $H:'/root/kc/*.log' $OUT/ 2>/dev/null
  log "$I ended: $END"
  destroy $I "session end: $END"
  log "credit \$$(credit) at end"; echo "$END" > $OUT/SESSION-END; exit 0
done
log "SESSION-END no-offer-passed"; echo no-offer > $OUT/SESSION-END

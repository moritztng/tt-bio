#!/bin/bash
# One unattended B200 session for the MOL1 fair B200 baseline (state/mol1/PROTOCOL.md), run DETACHED
# on pc. Rents the first B200 offer that passes (ssh up, power limit in window), pushes the MOL1 harness, runs
# box_mol1.sh on the box (which freezes the inputs itself), pulls /root/kc/mol1 every 3 minutes, destroys the box at MOL1-DONE/FAILED or the cap.
#   usage: mol1_session.sh <label> <cap_h> ['<vastai search query>'] [max_attempts]
# Status on stdout (redirect it); results perf/kitcmp/results/<label>/mol1/...
# Derived from bc4_session.sh, which proved the vast mechanics on 2026-10-19 (two routes to the box, the tried-offer
# file on disk, the credit floor, destroy-on-cap).
set -u
L=${1:?usage: mol1_session.sh <label> <cap_h> [query] [attempts]} CAP_H=${2:?cap hours} NMAX=${4:-6}
# reliability>=0.97 was the only clause that rejected a live B200 on 2026-10-19 (offer 51577286, 0.995 passes but the
# market is thin enough that one filter empties it); 0.90 keeps the junk out without emptying the list.
Q=${3:-'gpu_name=B200 num_gpus=1 cuda_max_good>=13.0 inet_down>=200 reliability>=0.90 dph<=7'}
MINW=${MINW:-990} MAXW=${MAXW:-1010} SLUG=${SLUG:-mol1-b200}
# Which script runs on the box.
BOX=${BOX:-box_mol1.sh}
P=$(cd "$(dirname "$0")" && pwd); V=~/.vast-venv/bin/vastai; B=/home/moritz/.coworker/state/vast-budget
HOLD=/home/moritz/.coworker/state/vast-holder; S4=/home/moritz/.coworker/state/mol1
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
  # The reaper only ever touches ids listed in OWNED, so an id that is gone must leave the list.
  sed -i "/^$1\$/d" $S4/gpu/OWNED 2>/dev/null
  sed -i "s|^$1  $CAP_H  $SLUG$|#$1 DESTROYED $(date -u +%FT%TZ) ($2), confirmed instances=$n|" $B; }
next_offer(){ $V search offers "$Q" -o dph --raw 2>/dev/null | python3 -c 'import json,sys
t=sys.argv[1].split()
print(next((str(o["id"]) for o in json.load(sys.stdin) if str(o["id"]) not in t), ""))' "$TRIED"; }

R(){ timeout 600 rsync -a --exclude '*.cif' -e "${S% root@*}" "$@"; }
# Pull every 3 minutes until MOL1-DONE/FAILED/stopped/cap, then destroy. CIFs stay on the box: the speed arms'
# structures are not evidence, and on 2026-10-31 pc's ~25 KB/s link hung one pull on 279 MB of them for 35 minutes,
# which froze this loop (no DONE check, no cap) while the box kept billing.
watch(){
END=""
while :; do
  sleep 180
  R $H:/root/kc/mol1/ $OUT/mol1/ 2>/dev/null; R $H:'/root/kc/*.log' $OUT/ 2>/dev/null
  st=$(timeout 30 $S 'ls /root/kc/mol1/MOL1-DONE' 2>/dev/null)
  [ -n "$st" ] && END=done
  # box_mol1.sh writes MOL1-FAILED when the gate trips or no arm produced a warm batch. Name it, so a
  # session that measured nothing never reads as "done" in this log (session 1, 2026-10-21).
  f=$(timeout 30 $S 'cat /root/kc/mol1/MOL1-FAILED 2>/dev/null' 2>/dev/null)
  [ -n "$f" ] && END="FAILED ($f)"
  [ -z "$st" ] && [ "$(state $I)" != running ] && END=stopped
  [ $(( $(date +%s) - T0 )) -ge $(python3 -c "print(int($CAP_H*3600-420))") ] && END=${END:-cap}
  [ -n "$END" ] && break
done
R $H:/root/kc/mol1/ $OUT/mol1/ 2>/dev/null; R $H:'/root/kc/*.log' $OUT/ 2>/dev/null
log "$I ended: $END"
destroy $I "session end: $END"
log "credit \$$(credit) at end"; echo "$END" > $OUT/SESSION-END; exit 0
}
# RESUME=1 re-attaches to the box this label already rents (a watcher that died or was stopped), without renting.
if [ -n "${RESUME:-}" ]; then
  read -r I O < $OUT/instance; S=$(cat $OUT/ssh); H=${S##* }
  T0=$($V show instance $I --raw | python3 -c 'import json,sys;print(int(json.load(sys.stdin)["start_date"]))')
  log "resumed watching $I (started $(date -u -d @$T0 +%FT%TZ))"; watch
fi

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
  # Arm the reaper in the same breath as the rental. Session 4 was armed by hand after launch, which is a
  # window in which a killed session leaves a $8.61/h box running with no owner and no timer.
  mkdir -p $S4/gpu; echo "$I" >> $S4/gpu/OWNED
  python3 -c "print(int($(date +%s) + $CAP_H*3600 + 1500))" > $S4/gpu/KEEP-UNTIL
  echo "# ^ $SLUG $L offer $O created $(date -u +%FT%TZ) by perf/kitcmp/mol1_session.sh" >> $B
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
  H=root@$HOST
  $S 'mkdir -p /root/kc' &&
    R $P/mol1_gpu.py $P/mol1_arms.py $P/mol1_inputs.py $P/$BOX $H:/root/kc/ &&
    echo "$I $O" | timeout 30 $S 'cat > /root/kc/instance' ||
    { destroy $I "push failed"; continue; }
  # BOXENV carries the box script's knobs across ssh (CELLS, FINAL_WARM, CUDA_VISIBLE_DEVICES). An env var
  # set around this script never reaches the remote shell, so without it a CELLS list silently ran the default.
  timeout 60 $S -f "cd /root/kc && setsid nohup env MOL1_SESSION=$L ${BOXENV:-} bash $BOX $CAP_H > box_mol1.log 2>&1 < /dev/null &" ||
    { destroy $I "launch failed"; continue; }
  log "$I running $BOX"
  watch
done
log "SESSION-END no-offer-passed"; echo no-offer > $OUT/SESSION-END

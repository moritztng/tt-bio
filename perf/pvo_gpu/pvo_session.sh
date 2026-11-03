#!/bin/bash
# One unattended A100-SXM4-40GB session for PVO (state/pvo/GPU.md), run DETACHED
# on pc. Rents the first A100-40 offer that passes (ssh up, power limit in window), pushes the PVO harness, runs
# box_pvo.sh on the box, pulls /root/kc/pvo every 3 minutes, destroys the box at PVO-DONE/FAILED or the cap.
#   usage: pvo_session.sh <label> <cap_h> ['<vastai search query>'] [max_attempts]
# Status on stdout (redirect it); results perf/pvo_gpu/results/<label>/pvo/...
# Derived from bc4_session.sh, which proved the vast mechanics on 2026-10-19 (two routes to the box, the tried-offer
# file on disk, the credit floor, destroy-on-cap).
set -u
L=${1:?usage: pvo_session.sh <label> <cap_h> [query] [attempts]} CAP_H=${2:?cap hours} NMAX=${4:-6}
# reliability>=0.97 was the only clause that rejected a live B200 on 2026-10-19 (offer 51577286, 0.995 passes but the
# market is thin enough that one filter empties it); 0.90 keeps the junk out without emptying the list.
Q=${3:-'gpu_name=A100_SXM4 gpu_ram<=45 num_gpus=1 cuda_max_good>=13.0 inet_down>=200 reliability>=0.90 dph<=1.0'}
MINW=${MINW:-400} MAXW=${MAXW:-410} SLUG=${SLUG:-pvo-a100}
# Which script runs on the box. INPUTS=<dir> pushes an already-frozen inputs tree (state/mol1/inputs), so the box
# reuses those MSAs instead of searching again. QB2=user@host:dir moves bulk data box<->qb2 directly (pc's link
# cannot carry it): qb2's key is authorized on the box, qb2 pushes dir/inputs to /root/kc/qinputs, and every pull
# also copies the box's /root/kc/pvo (with CIFs) to qb2 dir/<label>/ref.
BOX=${BOX:-box_pvo.sh}
P=$(cd "$(dirname "$0")" && pwd); V=~/.vast-venv/bin/vastai; B=/home/moritz/.coworker/state/vast-budget
HOLD=/home/moritz/.coworker/state/vast-holder; S4=/home/moritz/.coworker/state/pvo
K=$(cat ~/.config/vastai/vast_api_key); API=https://console.vast.ai/api/v0; FLOOR=${VAST_FLOOR_USD:-1}
OUT=$P/results/$L; mkdir -p "$OUT"
# Offers we have already paid for and lost live on disk, not in this process: a relaunch after a crash used to
# re-pick the offer it had just destroyed (2026-10-19, offer 50255941 twice, ~$2.8).
TRIEDF=$OUT/tried; touch "$TRIEDF"; TRIED=" $(tr '\n' ' ' < "$TRIEDF")"
# Offers that rented but never came up on ssh, fleet-wide (the 502559xx family, four times on 11-01).
TRIED="$TRIED$(tr '\n' ' ' < /home/moritz/.coworker/state/vast-bad-offers 2>/dev/null) "
log(){ echo "$(date -u +%FT%TZ) $*"; }
# A vast API blip returns an empty body; read as no credit it REFUSED and killed session mol1-b200-ref6r (11-01).
credit(){ local c n; for n in 1 2 3 4 5; do
  c=$(curl -s -H "Authorization: Bearer $K" "$API/users/current/" | python3 -c 'import json,sys;print(json.load(sys.stdin)["credit"])' 2>/dev/null)
  [ -n "$c" ] && { echo "$c"; return; }; sleep 20; done; echo 0; }
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

# qb2 -> box: vast's ssh proxy authenticates against keys ATTACHED to the instance, so a line appended to the box's
# authorized_keys alone was refused (session mol1-b200-ref1, 2026-11-01: gate passed, inputs never arrived, $5 spent).
# Attach through the API, append as well for the direct port, and prove qb2 gets in before the box is launched.
qb2_key(){ local k; k=$($Q2 'cat ~/.ssh/id_ed25519.pub') || return 1
  $V attach ssh $I "$k" >/dev/null 2>&1; echo "$k" | timeout 30 $S 'cat >> /root/.ssh/authorized_keys' || return 1
  for _ in 1 2 3 4 5 6; do timeout 60 $Q2 "${S% root@*} $H true" >/dev/null 2>&1 && return 0; sleep 30; done
  log "$I: qb2 cannot ssh to the box"; return 1; }
R(){ timeout 600 rsync -a --exclude '*.cif' --exclude 'ref/*/' -e "${S% root@*}" "$@"; }
# A fresh box's sshd can drop the connection right after the first login answered (mol1-b200-ref9, 11-01: the push
# died on "Connection reset" 40 s after boot and the only B200 offer went on the tried list), so the push retries.
push(){ local n; for n in 1 2 3 4; do R "$@" && return 0; sleep 20; done; return 1; }
# Pull every 3 minutes until PVO-DONE/FAILED/stopped/cap, then destroy. CIFs stay on the box: the speed arms'
# structures are not evidence, and on 2026-10-31 pc's ~25 KB/s link hung one pull on 279 MB of them for 35 minutes,
# which froze this loop (no DONE check, no cap) while the box kept billing.
QB2=${QB2:-}; Q2="ssh -o BatchMode=yes -o ConnectTimeout=15 ${QB2%%:*}"
offload(){ [ -z "$QB2" ] && return 0
  timeout 1800 $Q2 "mkdir -p ${QB2#*:}/$L && rsync -a -e '${S% root@*}' $H:/root/kc/pvo ${QB2#*:}/$L/" 2>/dev/null; }
watch(){
END=""
while :; do
  sleep 180
  R $H:/root/kc/pvo/ $OUT/pvo/ 2>/dev/null; R $H:'/root/kc/*.log' $OUT/ 2>/dev/null; offload
  st=$(timeout 30 $S 'ls /root/kc/pvo/PVO-DONE' 2>/dev/null)
  [ -n "$st" ] && END=done
  # box_pvo.sh writes PVO-FAILED when the gate trips or no arm produced a warm batch. Name it, so a
  # session that measured nothing never reads as "done" in this log (session 1, 2026-10-21).
  f=$(timeout 30 $S 'cat /root/kc/pvo/PVO-FAILED 2>/dev/null' 2>/dev/null)
  [ -n "$f" ] && END="FAILED ($f)"
  [ -z "$st" ] && [ "$(state $I)" != running ] && END=stopped
  [ $(( $(date +%s) - T0 )) -ge $(python3 -c "print(int($CAP_H*3600-420))") ] && END=${END:-cap}
  [ -n "$END" ] && break
done
R $H:/root/kc/pvo/ $OUT/pvo/ 2>/dev/null; R $H:'/root/kc/*.log' $OUT/ 2>/dev/null; offload
log "$I ended: $END"
destroy $I "session end: $END"
log "credit \$$(credit) at end"; echo "$END" > $OUT/SESSION-END; exit 0
}
# RESUME=1 re-attaches to the box this label already rents (a watcher that died or was stopped), without renting.
if [ -n "${RESUME:-}" ]; then
  read -r I O < $OUT/instance; S=$(cat $OUT/ssh); H=${S##* }
  T0=$($V show instance $I --raw | python3 -c 'import json,sys;print(int(json.load(sys.stdin)["start_date"]))')
  # The guard keys ownership on this pid. Without the rewrite the holder still names the replaced watcher,
  # and vast_guard.sh stopped both 11-01 reference boxes mid-run the first time no worker turn was alive.
  mkdir -p $HOLD; echo $$ > $HOLD/$I
  log "resumed watching $I (started $(date -u -d @$T0 +%FT%TZ))"; watch
fi

MAXDPH=$(echo "$Q" | grep -oP 'dph<=\K[0-9.]+'); C=$(credit)
python3 -c "import sys; sys.exit(0 if $C - $CAP_H * ${MAXDPH:-1} >= $FLOOR else 1)" ||
  { log "REFUSED: credit \$$C - $CAP_H h x \$${MAXDPH:-1}/h would cross the \$$FLOOR floor"; exit 4; }
log "credit \$$C at start, cap $CAP_H h"

for _ in $(seq $NMAX); do
  O=$(next_offer); [ -z "$O" ] && { log "no untried offer matches: $Q"; break; }
  TRIED="$TRIED$O "; echo "$O" >> "$TRIEDF"
  C=$(credit); python3 -c "import sys; sys.exit(0 if $C - $CAP_H * ${MAXDPH:-1} >= $FLOOR else 1)" ||
    { log "REFUSED at offer $O: credit \$$C would cross the \$$FLOOR floor"; exit 4; }
  echo "-  $CAP_H  $SLUG  # $L creating on offer $O" >> $B
  I=$($V create instance $O --image nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 --disk ${DISK:-150} --ssh --direct --label $SLUG-$L --raw 2>&1 | grep -oP "new_contract['\"]:\s*\K[0-9]+")
  [ -z "$I" ] && { sed -i '$d' $B; log "offer $O: create failed"; continue; }
  mkdir -p $HOLD; echo $$ > $HOLD/$I; sed -i "\$s|.*|$I  $CAP_H  $SLUG|" $B
  # Arm the reaper in the same breath as the rental. Session 4 was armed by hand after launch, which is a
  # window in which a killed session leaves a $8.61/h box running with no owner and no timer.
  mkdir -p $S4/gpu; echo "$I" >> $S4/gpu/OWNED
  python3 -c "print(int($(date +%s) + $CAP_H*3600 + 1500))" > $S4/gpu/KEEP-UNTIL
  echo "# ^ $SLUG $L offer $O created $(date -u +%FT%TZ) by perf/pvo_gpu/pvo_session.sh" >> $B
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
    push $P/pvo_fold.py $P/pick.py $P/burn_gate.py $P/recipe-opendde.json $P/recipe-mol1.json $P/$BOX ${EXTRA_FILES:-} $H:/root/kc/ &&
    { [ -z "${INPUTS:-}" ] || R "$INPUTS/" $H:/root/kc/inputs/; } &&
    # RFD4's target is a .cif, which R() excludes (it keeps pvo-accuracy's reference structures off the box),
    # so the RFD4 input tree gets its own push.
    { [ -z "${RFD4IN:-}" ] || timeout 600 rsync -a -e "${S% root@*}" "$RFD4IN/" $H:/root/kc/rfd4in/; } &&
    echo "$I $O" | timeout 30 $S 'cat > /root/kc/instance' &&
    { [ -z "$QB2" ] || { qb2_key && { $Q2 "rsync -a -e '${S% root@*}' ${QB2#*:}/inputs/ $H:/root/kc/qinputs/ && ${S% root@*} $H touch /root/kc/QINPUTS-DONE" \
        > $OUT/qinputs.log 2>&1 & }; }; } ||
    { destroy $I "push failed"; continue; }
  # BOXENV carries the box script's knobs across ssh (CELLS, FINAL_WARM, CUDA_VISIBLE_DEVICES). An env var
  # set around this script never reaches the remote shell, so without it a CELLS list silently ran the default.
  timeout 60 $S -f "cd /root/kc && setsid nohup env PVO_SESSION=$L ${BOXENV:-} bash $BOX $CAP_H > box_pvo.log 2>&1 < /dev/null &" ||
    { destroy $I "launch failed"; continue; }
  log "$I running $BOX"
  watch
done
log "SESSION-END no-offer-passed"; echo no-offer > $OUT/SESSION-END

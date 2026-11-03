#!/bin/bash
# One unattended PNR GPU session (state/pnr/PROTOCOL.md), run DETACHED on pc: rents one box of the named GPU, pushes
# the three harnesses and the frozen inputs, runs box_pnr.sh (Mol-1, BindCraft 2, RFD4 in one session), pulls
# /root/kc every 3 minutes, destroys the box at PNR-DONE / stopped / the cap and confirms it gone over REST.
#   usage: pnr_session.sh --gpu a100|h200|b200|dry <label> <cap_h> ['<vastai search query>'] [max_attempts]
#   WAIT_H=<h>: no matching offer yet (the single-B200 market is often empty) -> look again every 10 min, that long.
# Results perf/pnr/gpu/<label>/ (structures stay on the box). Status on stdout (redirect it).
# Rental mechanics are mol1_session.sh's (two ssh routes, tried-offer file, fleet bad-offer list, credit retries,
# push retries, destroy-on-cap), plus session.sh's download check. Refuses to rent if credit - cap x max $/h < $20.
set -u
[ "${1:-}" = --gpu ] || { echo "usage: pnr_session.sh --gpu a100|h200|b200|dry <label> <cap_h> [query] [attempts]" >&2; exit 2; }
GPU=$2; shift 2
# offer filter, power-limit window (W) and BindCraft 2 SM clock floor (MHz, ~90 % of the card's boost) per GPU.
# h200: gpu_name=H200 is the SXM part only (the 600 W NVL is "H200 NVL"); a100: the 400 W SXM4 80 GB, not the 500 W HGX.
case $GPU in
  a100) DQ='gpu_name=A100_SXM4 num_gpus=1 gpu_ram>=79 cuda_max_good>=13.0 inet_down>=200 reliability>=0.95 dph<=1.6' W0=390 W1=400 SM=1250;;
  h200) DQ='gpu_name=H200 num_gpus=1 cuda_max_good>=13.0 inet_down>=200 reliability>=0.95 dph<=6' W0=690 W1=700 SM=1750;;
  b200) DQ='gpu_name=B200 num_gpus=1 cuda_max_good>=13.0 inet_down>=200 reliability>=0.90 dph<=7' W0=990 W1=1010 SM=1800;;
  dry)  DQ='gpu_name=RTX_4090 num_gpus=1 cuda_max_good>=13.0 inet_down>=400 reliability>=0.95 dph<=0.6' W0=0 W1=10000 SM=0;;
  *) echo "pnr_session.sh: unknown --gpu $GPU" >&2; exit 2;;
esac
L=${1:?label} CAP_H=${2:?cap hours} Q=${3:-$DQ} NMAX=${4:-6}
MINW=${MINW:-$W0} MAXW=${MAXW:-$W1} SLUG=${SLUG:-pnr-gpu} FLOOR=${VAST_FLOOR_USD:-20}
BOXENV="PNR_LABEL=$L MIN_SM=$SM ${BOXENV:-}"; [ $GPU = dry ] && BOXENV="DRY=1 $BOXENV"
P=$(cd "$(dirname "$0")" && pwd); KC=$P/../kitcmp; V=~/.vast-venv/bin/vastai
B=/home/moritz/.coworker/state/vast-budget HOLD=/home/moritz/.coworker/state/vast-holder
K=$(cat ~/.config/vastai/vast_api_key); API=https://console.vast.ai/api/v0
IMAGE=nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04
OUT=$P/gpu/$L; mkdir -p "$OUT"; python3 $P/pnr_inputs.py $OUT/in > /dev/null || exit 3
TRIEDF=$OUT/tried; touch "$TRIEDF"
TRIED=" $(tr '\n' ' ' < "$TRIEDF") $(tr '\n' ' ' < /home/moritz/.coworker/state/vast-bad-offers 2>/dev/null) "
log(){ echo "$(date -u +%FT%TZ) $*"; }
credit(){ local c n; for n in 1 2 3 4 5; do
  c=$(curl -s -H "Authorization: Bearer $K" "$API/users/current/" | python3 -c 'import json,sys;print(json.load(sys.stdin)["credit"])' 2>/dev/null)
  [ -n "$c" ] && { echo "$c"; return; }; sleep 20; done; echo 0; }
gone(){ curl -s -H "Authorization: Bearer $K" "$API/instances/?owner=me" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["instances"]))'; }
state(){ curl -s -H "Authorization: Bearer $K" "$API/instances/?owner=me" | python3 -c 'import json,sys
print(next((x.get("intended_status") or "" for x in json.load(sys.stdin)["instances"] if x["id"]==int(sys.argv[1])), "gone"))' $1; }
destroy(){ curl -s -X DELETE -H "Authorization: Bearer $K" "$API/instances/$1/" >/dev/null; sleep 10
  local n s; n=$(gone); s=$(state $1); log "DESTROYED $1 ($2); this instance over REST: $s; instances left: $n"; rm -f $HOLD/$1
  echo "$1 $(date -u +%FT%TZ) $s" >> $OUT/destroyed
  sed -i "s|^$1  $CAP_H  $SLUG$|#$1 DESTROYED $(date -u +%FT%TZ) ($2), REST: $s, instances=$n|" $B; }
next_offer(){ $V search offers "$Q" -o dph --raw 2>/dev/null | python3 -c 'import json,sys
t=sys.argv[1].split()
print(next((str(o["id"]) for o in json.load(sys.stdin) if str(o["id"]) not in t), ""))' "$TRIED"; }
R(){ timeout 900 rsync -a --exclude '*.cif' --exclude '*.cif.gz' --exclude '*.pdb' --exclude '*.png' --exclude '*.pt' \
  --exclude '*.npz' --exclude 'bin/' -e "${S% root@*}" "$@"; }
push(){ local n; for n in 1 2 3 4; do R "$@" && return 0; sleep 20; done; return 1; }

watch(){
END=""
while :; do
  sleep 180
  R $H:/root/kc/ $OUT/box/ 2>/dev/null
  [ -n "$(timeout 30 $S 'ls /root/kc/PNR-DONE' 2>/dev/null)" ] && END=done
  [ -z "$END" ] && [ "$(state $I)" != running ] && END=stopped
  [ $(( $(date +%s) - T0 )) -ge $(python3 -c "print(int($CAP_H*3600-420))") ] && END=${END:-cap}
  [ -n "$END" ] && break
done
R $H:/root/kc/ $OUT/box/ 2>/dev/null
log "$I ended: $END ($(tr '\n' ';' < $OUT/box/PNR-PROGRESS 2>/dev/null))"
destroy $I "session end: $END"
log "credit \$$(credit) at end"; echo "$END" > $OUT/SESSION-END; exit 0
}
# RESUME=1 re-attaches to the box this label already rents (a watcher that died), without renting.
if [ -n "${RESUME:-}" ]; then
  read -r I O < $OUT/instance; S=$(cat $OUT/ssh); H=${S##* }
  T0=$($V show instance $I --raw | python3 -c 'import json,sys;print(int(json.load(sys.stdin)["start_date"]))')
  mkdir -p $HOLD; echo $$ > $HOLD/$I; log "resumed watching $I"; watch
fi

MAXDPH=$(echo "$Q" | grep -oP 'dph<=\K[0-9.]+')
fits(){ python3 -c "import sys; sys.exit(0 if $1 - $CAP_H * $MAXDPH >= $FLOOR else 1)"; }
C=$(credit); fits $C || { log "REFUSED: credit \$$C - $CAP_H h x \$$MAXDPH/h would cross the \$$FLOOR floor"; exit 4; }
log "credit \$$C at start, gpu $GPU, cap $CAP_H h"; echo "$C" > $OUT/credit-start

TW=$(date +%s)
for _ in $(seq $NMAX); do
  O=$(next_offer)
  while [ -z "$O" ] && [ $(( $(date +%s) - TW )) -lt $(python3 -c "print(int(${WAIT_H:-0}*3600))") ]; do sleep 600; O=$(next_offer); done
  [ -z "$O" ] && { log "no untried offer matches: $Q"; break; }
  TRIED="$TRIED$O "; echo "$O" >> "$TRIEDF"
  C=$(credit); fits $C || { log "REFUSED at offer $O: credit \$$C would cross the \$$FLOOR floor"; exit 4; }
  echo "-  $CAP_H  $SLUG  # $L creating on offer $O" >> $B
  I=$($V create instance $O --image $IMAGE --disk ${DISK:-200} --ssh --direct --label $SLUG-$L --raw 2>&1 | grep -oP "new_contract['\"]:\s*\K[0-9]+")
  [ -z "$I" ] && { sed -i '$d' $B; log "offer $O: create failed"; continue; }
  mkdir -p $HOLD; echo $$ > $HOLD/$I; sed -i "\$s|.*|$I  $CAP_H  $SLUG|" $B
  echo "# ^ $SLUG $L offer $O created $(date -u +%FT%TZ) by perf/pnr/pnr_session.sh" >> $B
  T0=$(date +%s); log "created $I on offer $O"; S=""
  while [ $(( $(date +%s) - T0 )) -lt 720 ]; do
    [ $(( $(date +%s) - T0 )) -ge 120 ] && [ "$(state $I)" != running ] && break
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
  BW=$(timeout 40 $S 'curl -s -o /dev/null -m 20 -w "%{speed_download}" https://download.pytorch.org/whl/cu130/torch-2.9.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl' 2>/dev/null | tail -1 | cut -d. -f1)
  log "$I download ${BW:-?} B/s"
  [ "${BW:-0}" -ge ${MINBW:-20000000} ] || { destroy $I "download ${BW:-?} B/s < ${MINBW:-20000000} B/s"; continue; }
  echo "$S" > $OUT/ssh; echo "$I $O" > $OUT/instance; H=root@$HOST
  $V show instance $I --raw 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); [d.pop(k) for k in list(d) if 'token' in k.lower()]; json.dump(d,sys.stdout,indent=1)" > $OUT/instance.json || true
  $S 'mkdir -p /root/kc' &&
    push $KC/box_mol1.sh $KC/mol1_gpu.py $KC/mol1_arms.py $KC/box_bc_setup.sh $KC/bc2_gpu.py $KC/bc2_traj.py $KC/bc4_arms.py \
      $KC/box_rfd4.sh $KC/rfd4_gpu.py $KC/rfd4_arms.py $P/box_pnr.sh $H:/root/kc/ &&
    push $OUT/in/ $H:/root/kc/inputs/ &&
    echo "$I $O $IMAGE" | timeout 30 $S 'cat > /root/kc/instance' ||
    { destroy $I "push failed"; continue; }
  timeout 60 $S -f "cd /root/kc && setsid nohup env $BOXENV bash box_pnr.sh $CAP_H > box_pnr.log 2>&1 < /dev/null &" ||
    { destroy $I "launch failed"; continue; }
  log "$I running box_pnr.sh ($BOXENV)"
  watch
done
log "SESSION-END no-offer-passed"; echo no-offer > $OUT/SESSION-END

#!/bin/bash
# One unattended GPU session for spd-kitcmp, run DETACHED on pc (derived from wk/pfm-gpu perf/pfm_gpu/session.sh).
# Rents the first offer that passes (ssh up, power limit >= MINW, >= 20 MB/s download, burn_gate.py not throttled),
# sets up and times every kit in MODELS one after another, pulls results every model, destroys the box and confirms
# it is gone. CAP_H hard-caps the billed time. Refuses to start if credit - CAP_H x max $/h would cross the $20 floor.
#   usage: session.sh <label> <cap_h> <minW> '<vastai search query>' [max_attempts]
#   e.g.   session.sh a 5.0 390 'gpu_name=A100_SXM4 num_gpus=1 gpu_ram>=79 cuda_max_good>=13.0 inet_down>=200 reliability>=0.98 dph<=1.2'
# Status: stdout (redirect it); results perf/kitcmp/results/<label>/<model>/...
set -u
L=$1 CAP_H=$2 MINW=$3 Q=$4 NMAX=${5:-8}; TRIED=" "; MODELS=${MODELS:-protenix_v2 boltz2 openfold3 opendde}
P=$(cd "$(dirname "$0")" && pwd); V=~/.vast-venv/bin/vastai; B=/home/moritz/.coworker/state/vast-budget
K=$(cat ~/.config/vastai/vast_api_key); API=https://console.vast.ai/api/v0; FLOOR=20
KIT=${KIT:-/home/moritz/scratch/kitsrc/uplifting-biomolecular-modeling}
OUT=$P/results/$L; mkdir -p $OUT
log(){ echo "$(date -u +%FT%TZ) $*"; }
credit(){ curl -s -H "Authorization: Bearer $K" "$API/users/current/" | python3 -c 'import json,sys;print(json.load(sys.stdin)["credit"])'; }
MAXDPH=$(echo "$Q" | grep -oP 'dph<=\K[0-9.]+'); C=$(credit)
python3 -c "import sys; sys.exit(0 if $C - $CAP_H * ${MAXDPH:-1.5} >= $FLOOR else 1)" ||
  { log "REFUSED: credit \$$C - $CAP_H h x \$${MAXDPH:-1.5}/h would cross the \$$FLOOR floor"; exit 4; }
log "credit \$$C at start"
gone(){ curl -s -H "Authorization: Bearer $K" "$API/instances/?owner=me" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["instances"]))'; }
destroy(){ # <id> <why>
  curl -s -X DELETE -H "Authorization: Bearer $K" "$API/instances/$1/" >/dev/null; sleep 10
  local n; n=$(gone); log "DESTROYED $1 ($2); instances left: $n"
  sed -i "s|^$1  $CAP_H  spd-kitcmp$|#$1 DESTROYED $(date -u +%FT%TZ) ($2), confirmed instances=$n|" $B
}
next_offer(){ $V search offers "$Q" -o dph --raw 2>/dev/null | python3 -c 'import json,sys
t=sys.argv[1].split()
print(next((str(o["id"]) for o in json.load(sys.stdin) if str(o["id"]) not in t), ""))' "$TRIED"; }
state(){ curl -s -H "Authorization: Bearer $K" "$API/instances/?owner=me" | python3 -c 'import json,sys
print(next((x.get("intended_status") or "" for x in json.load(sys.stdin)["instances"] if x["id"]==int(sys.argv[1])), "gone"))' $1; }
python3 $P/make_inputs.py $OUT/in > /dev/null
tar -czf $OUT/kit.tgz -C "$KIT" common $MODELS
for _ in $(seq $NMAX); do
  O=$(next_offer); [ -z "$O" ] && { log "no offer matches: $Q"; break; }; TRIED="$TRIED$O "
  echo "-  $CAP_H  spd-kitcmp  # $L creating on offer $O" >> $B
  I=$($V create instance $O --image nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 --disk 150 --ssh --direct --label spd-kitcmp-$L --raw 2>&1 | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["new_contract"])
except Exception: print("")')
  if [ -z "$I" ]; then sed -i '$d' $B; log "offer $O: create failed"; continue; fi
  sed -i "\$s|.*|$I  $CAP_H  spd-kitcmp|" $B; echo "# ^ spd-kitcmp $L offer $O created $(date -u +%FT%TZ) by perf/kitcmp/session.sh" >> $B
  T0=$(date +%s); log "created $I on offer $O"; S=""
  while [ $(( $(date +%s) - T0 )) -lt 1500 ]; do
    [ $(( $(date +%s) - T0 )) -ge 120 ] && [ "$(state $I)" != running ] && break
    read -r IP PORT < <($V show instance $I --raw 2>/dev/null | python3 -c 'import json,sys
x=json.load(sys.stdin); p=(x.get("ports") or {}).get("22/tcp")
print(x.get("public_ipaddr","").strip(), p[0]["HostPort"] if p else "") if x.get("actual_status")=="running" else print("","")' 2>/dev/null)
    if [ -n "${PORT:-}" ]; then
      S="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -p $PORT root@$IP"
      timeout 30 $S true >/dev/null 2>&1 && break; S=""
    fi
    sleep 30
  done
  [ -z "$S" ] && { destroy $I "no ssh: intended_status=$(state $I)"; continue; }
  W=$(timeout 30 $S 'nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits' 2>/dev/null | tail -1 | cut -d. -f1)
  log "$I up at $IP:$PORT, power limit ${W:-?} W"
  [ "${W:-0}" -ge "$MINW" ] || { destroy $I "power limit ${W:-?} W < $MINW"; continue; }
  BW=$(timeout 40 $S 'curl -s -o /dev/null -m 20 -w "%{speed_download}" https://download.pytorch.org/whl/cu130/torch-2.9.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl' 2>/dev/null | tail -1 | cut -d. -f1)
  log "$I download ${BW:-?} B/s"
  [ "${BW:-0}" -ge 20000000 ] || { destroy $I "download ${BW:-?} B/s < 20 MB/s"; continue; }
  echo "$S" > $OUT/ssh; echo "$I $O" > $OUT/instance
  R(){ rsync -a -e "${S% root@*}" "$@"; }; H=root@$IP
  $S 'mkdir -p /root/kit /root/kc/in /root/kc/results' && R $OUT/kit.tgz $H:/root/ && $S 'tar -xzf /root/kit.tgz -C /root/kit' &&
    R $OUT/in/ $H:/root/kc/in/ && R $P/box_setup.sh $P/box_arms.sh $P/burn_gate.py $P/kctime.py $H:/root/kc/ || { destroy $I "push failed"; continue; }
  $S "cd /root/kc && cat > chain.sh <<'X'
for m in $MODELS; do
  bash box_setup.sh \$m > setup-\$m.log 2>&1 || { echo \$m setup-failed >> CHAIN-PROGRESS; continue; }
  # burn gate once, in the first venv that has torch (a hot host is rejected before any timed arm)
  [ -f burn.log ] || /root/kit/\$m/venv/bin/python burn_gate.py > burn.log 2>&1 || { echo THERMAL-FAIL > THERMAL-FAIL; exit 3; }
  bash box_arms.sh \$m > arms-\$m.log 2>&1
  echo \$m \$? >> CHAIN-PROGRESS
done
echo done > CHAIN-DONE
X
setsid nohup bash chain.sh > chain.log 2>&1 < /dev/null &"
  END=""
  while :; do
    sleep 300
    R $H:/root/kc/results/ $OUT/results/ 2>/dev/null; R $H:'/root/kc/*.log' $H:/root/kc/CHAIN-PROGRESS $OUT/ 2>/dev/null
    st=$(timeout 30 $S 'ls /root/kc/CHAIN-DONE /root/kc/THERMAL-FAIL' 2>/dev/null)
    case "$st" in *THERMAL-FAIL*) END=thermal;; *CHAIN-DONE*) END=done;; esac
    [ -z "$st" ] && [ "$(state $I)" != running ] && END=stopped
    [ $(( $(date +%s) - T0 )) -ge $(python3 -c "print(int($CAP_H*3600-600))") ] && END=${END:-cap}
    [ -n "$END" ] && break
  done
  log "$I ended: $END ($(cat $OUT/CHAIN-PROGRESS 2>/dev/null | tr '\n' ' '))"
  R $H:/root/kc/results/ $OUT/results/ 2>/dev/null; R $H:'/root/kc/*.log' $OUT/ 2>/dev/null
  destroy $I "session end: $END"; log "credit \$$(credit) at end"
  [ "$END" = stopped ] && { mv $OUT/results $OUT/results.stopped-$I; continue; }
  [ "$END" = thermal ] && { mv $OUT/burn.log $OUT/burn-hot-$I.log; rm -rf $OUT/results; continue; }
  echo "$END" > $OUT/SESSION-END; log "SESSION-END $END"; exit 0
done
log "SESSION-END no-offer-passed"; echo no-offer > $OUT/SESSION-END

#!/bin/bash
# One unattended GPU session, run DETACHED on pc. Rents the first offer that passes, runs everything, pulls the
# results, destroys the box and confirms it is gone. A host is rejected (destroyed, next offer) if ssh is not up in
# 30 min, its power limit is under MINW, or burn_gate.py sees thermal throttle. CAP_H hard-caps the billed time of
# an accepted box: past it the results so far are pulled and the box is destroyed regardless.
# Offers are searched live before each attempt (a pre-listed offer is often already rented: vast then queues the
# instance as intended_status=stopped / resources_unavailable, which is dropped after 2 min).
#   usage: session.sh <label> <cap_h> <minW> <with_acc 0|1> '<vastai search query>' [max_attempts]
# Status: /tmp/pfmgpu/<label>.log; results perf/pfm_gpu/results/<label>/; accuracy -> ~/pfm-accuracy-data/gpu/out/.
set -u
L=$1 CAP_H=$2 MINW=$3 ACC=$4 Q=$5 NMAX=${6:-8}; TRIED=" "
P=$(cd "$(dirname "$0")" && pwd); V=~/.vast-venv/bin/vastai; B=/home/moritz/.coworker/state/vast-budget
K=$(cat ~/.config/vastai/vast_api_key); API=https://console.vast.ai/api/v0
OUT=$P/results/$L; mkdir -p $OUT
log(){ echo "$(date -u +%FT%TZ) $*"; }
gone(){ curl -s -H "Authorization: Bearer $K" "$API/instances/?owner=me" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["instances"]))'; }
destroy(){ # <id> <why>
  curl -s -X DELETE -H "Authorization: Bearer $K" "$API/instances/$1/" >/dev/null; sleep 10
  local n; n=$(gone); log "DESTROYED $1 ($2); instances left: $n"
  sed -i "s|^$1  $CAP_H  pfm-gpu$|#$1 DESTROYED $(date -u +%FT%TZ) ($2), confirmed instances=$n|" $B
}
next_offer(){ $V search offers "$Q" -o dph --raw 2>/dev/null | python3 -c 'import json,sys
t=sys.argv[1].split()
print(next((str(o["id"]) for o in json.load(sys.stdin) if str(o["id"]) not in t), ""))' "$TRIED"; }
state(){ curl -s -H "Authorization: Bearer $K" "$API/instances/?owner=me" | python3 -c 'import json,sys
print(next((x.get("intended_status") or "" for x in json.load(sys.stdin)["instances"] if x["id"]==int(sys.argv[1])), "gone"))' $1; }
for _ in $(seq $NMAX); do
  O=$(next_offer); [ -z "$O" ] && { log "no offer matches: $Q"; break; }; TRIED="$TRIED$O "
  echo "-  $CAP_H  pfm-gpu  # $L creating on offer $O" >> $B
  I=$($V create instance $O --image nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 --disk 80 --ssh --direct --label pfm-gpu-$L --raw 2>&1 | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["new_contract"])
except Exception: print("")')
  if [ -z "$I" ]; then sed -i '$d' $B; log "offer $O: create failed"; continue; fi
  sed -i "\$s|.*|$I  $CAP_H  pfm-gpu|" $B; echo "# ^ $L offer $O created $(date -u +%FT%TZ) by session.sh" >> $B
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
  [ -z "$S" ] && { destroy $I "no ssh: intended_status=$(state $I) after $(( ($(date +%s) - T0) / 60 )) min"; continue; }
  W=$(timeout 30 $S 'nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits' 2>/dev/null | tail -1 | cut -d. -f1)
  log "$I up at $IP:$PORT, power limit ${W:-?} W"
  [ "${W:-0}" -ge "$MINW" ] || { destroy $I "power limit ${W:-?} W < $MINW"; continue; }
  echo "$S" > $OUT/ssh; echo "$I $O" > $OUT/instance
  R(){ rsync -a -e "${S% root@*}" "$@"; }; H=root@$IP
  $S 'mkdir -p /root/kit /root/pfm/in /root/pfm/msa /root/pfm/acc /weights/protenix/checkpoint' &&
  R /tmp/pfmgpu/kit.tgz $H:/root/ && $S 'tar -xzf /root/kit.tgz -C /root/kit' &&
  R $P/msa/ $H:/root/pfm/msa/ && R $P/in_timing.json $H:/root/pfm/in/timing.json &&
  R $P/box_setup.sh $P/box_arms.sh $P/burn_gate.py $H:/root/pfm/ &&
  { [ "$ACC" = 0 ] || R /home/moritz/pfm-accuracy-data/box/ $H:/root/pfm/acc/; } &&
  R /home/moritz/.boltz/protenix-v2.pt $H:/weights/protenix/checkpoint/ || { destroy $I "push failed"; continue; }
  log "$I pushed; starting setup + arms"
  $S 'cd /root/pfm && cat > chain.sh <<"E"
bash box_setup.sh > setup.log 2>&1; grep -q "^SETUP-OK" setup.log || { echo SETUP-FAILED > CHAIN-FAILED; exit 1; }
bash box_arms.sh > arms.log 2>&1; echo $? > arms.rc
E
setsid nohup bash chain.sh > chain.log 2>&1 < /dev/null &'
  END=""
  while :; do
    sleep 60
    st=$(timeout 30 $S 'cd /root/pfm; ls results/ARMS-DONE results/THERMAL-FAIL CHAIN-FAILED arms.rc 2>/dev/null' 2>/dev/null | tr '\n' ' ')
    case "$st" in *THERMAL-FAIL*) END=thermal;; *ARMS-DONE*) END=done;; *CHAIN-FAILED*|*arms.rc*) END=failed;; esac
    [ $(( $(date +%s) - T0 )) -ge $(python3 -c "print(int($CAP_H*3600-900))") ] && END=${END:-cap}
    [ -n "$END" ] && break
  done
  log "$I ended: $END"
  R $H:/root/pfm/results/ $OUT/ ; R $H:/root/pfm/setup.log $H:/root/pfm/arms.log $H:/root/pfm/chain.log $OUT/ 2>/dev/null
  if [ "$ACC" = 1 ]; then mkdir -p /home/moritz/pfm-accuracy-data/gpu/out; R $H:/root/pfm/acc/out/ /home/moritz/pfm-accuracy-data/gpu/out/; log "acc pulled: $(find /home/moritz/pfm-accuracy-data/gpu/out -name '*.cif' | wc -l) cif"; fi
  destroy $I "session end: $END"
  [ "$END" = thermal ] && { log "$I rejected hot: $(cat $OUT/burn.log 2>/dev/null | tail -1)"; mv $OUT $OUT.hot-$I; mkdir -p $OUT; continue; }
  echo "$END" > $OUT/SESSION-END; log "SESSION-END $END"; exit 0
done
log "SESSION-END no-offer-passed"; echo no-offer > $OUT/SESSION-END

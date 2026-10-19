#!/bin/bash
# One unattended GPU session that times the Anthropic kit (and the stock arms) on the tt-bio.com perf-page fixtures, run DETACHED on pc.
# Rents the first offer that passes (ssh up, MINW <= power limit <= MAXW (per GPU, env overrides), >= 20 MB/s download, burn_gate.py not throttled),
# sets up and times every kit in MODELS one after another, pulls results every model, destroys the box and confirms
# it is gone. CAP_H hard-caps the billed time. Refuses to start if credit - CAP_H x max $/h would cross the $20 floor.
#   usage: session.sh --gpu a100|h200|b200 <label> <cap_h> ['<vastai search query>'] [max_attempts]
#   WAIT_H=<h>: keep looking for a matching offer that long (every 10 min) before giving up.
#   The GPU picks the offer filter, the power-limit window (MINW/MAXW, env overrides), the kit's --config and the
#   TORCH_CUDA_ARCH_LIST every native build on the box uses; the query argument replaces the default filter.
# Status: stdout (redirect it); results perf/kitcmp/results/<label>/<model>/... (<model>/box = the instance it ran on).
# A box the host stops mid-session: its finished models are kept and the next box runs only the rest.
set -u
[ "${1:-}" = --gpu ] || { echo "usage: session.sh --gpu a100|h200|b200 <label> <cap_h> ['<query>'] [max_attempts]" >&2; exit 2; }
GPU=$2; shift 2
# kit config / arch / default offer filter / power window (W) per GPU; the kit has no b200 config, h200 is its nearest
case $GPU in
  a100) CFG=a100 ARCH=8.0  DQ='gpu_name=A100_SXM4 num_gpus=1 gpu_ram>=79 cuda_max_good>=13.0 inet_down>=200 reliability>=0.98 dph<=1.2' W0=390 W1=400;;
  h200) CFG=h200 ARCH=9.0  DQ='gpu_name=H200 num_gpus=1 cuda_max_good>=13.0 inet_down>=200 reliability>=0.97 dph<=5.5' W0=690 W1=700;;
  b200) CFG=h200 ARCH=10.0 DQ='gpu_name=B200 num_gpus=1 cuda_max_good>=13.0 inet_down>=200 dph<=10' W0=990 W1=1000;;
  *) echo "session.sh: unknown --gpu $GPU (a100|h200|b200)" >&2; exit 2;;
esac
L=$1 CAP_H=$2 Q=${3:-$DQ} NMAX=${4:-8}; MINW=${MINW:-$W0} MAXW=${MAXW:-$W1}; TRIED=" "; SLUG=${SLUG:-bmk-$GPU}
KC_N=${KC_N:-5}  # folds per arm: 1 cold + KC_N-1 warm
MODELS=${MODELS:-protenix_v2 boltz2 openfold3 opendde}
P=$(cd "$(dirname "$0")" && pwd); V=~/.vast-venv/bin/vastai; B=/home/moritz/.coworker/state/vast-budget
K=$(cat ~/.config/vastai/vast_api_key); API=https://console.vast.ai/api/v0; FLOOR=20
KIT=${KIT:-/home/moritz/scratch/kitsrc-bmk/uplifting-biomolecular-modeling}
PKL=/home/moritz/.cache/opendde/common/components.cif.rdkit_mol.pkl PXCKPT=/home/moritz/.boltz/protenix-v2.pt
HOLD=/home/moritz/.coworker/state/vast-holder  # <id> -> this pid: vast_guard leaves a box alone while its session lives
PXV1=/home/moritz/.coworker/protenix-ref-data/protenix_base_default_v0.5.0.pt  # sha256 9ea20b0a = tt-bio protenix-v1
OUT=$P/results/$L; mkdir -p $OUT
printf "KC_GPU=%s KC_CFG=%s KC_ARCH=%s KC_N=%s\n" $GPU $CFG $ARCH $KC_N > $OUT/gpu.env; git -C "$KIT" rev-parse HEAD > $OUT/kit-commit
log(){ echo "$(date -u +%FT%TZ) $*"; }
credit(){ curl -s -H "Authorization: Bearer $K" "$API/users/current/" | python3 -c 'import json,sys;print(json.load(sys.stdin)["credit"])'; }
MAXDPH=$(echo "$Q" | grep -oP 'dph<=\K[0-9.]+'); C=$(credit)
python3 -c "import sys; sys.exit(0 if $C - $CAP_H * ${MAXDPH:-1.5} >= $FLOOR else 1)" ||
  { log "REFUSED: credit \$$C - $CAP_H h x \$${MAXDPH:-1.5}/h would cross the \$$FLOOR floor"; exit 4; }
log "credit \$$C at start"; TW=$(date +%s)
gone(){ curl -s -H "Authorization: Bearer $K" "$API/instances/?owner=me" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["instances"]))'; }
destroy(){ # <id> <why>
  curl -s -X DELETE -H "Authorization: Bearer $K" "$API/instances/$1/" >/dev/null; sleep 10
  local n; n=$(gone); log "DESTROYED $1 ($2); instances left: $n"
  rm -f $HOLD/$1; sed -i "s|^$1  $CAP_H  $SLUG$|#$1 DESTROYED $(date -u +%FT%TZ) ($2), confirmed instances=$n|" $B
}
next_offer(){ $V search offers "$Q" -o dph --raw 2>/dev/null | python3 -c 'import json,sys
t=sys.argv[1].split()
print(next((str(o["id"]) for o in json.load(sys.stdin) if str(o["id"]) not in t), ""))' "$TRIED"; }
state(){ curl -s -H "Authorization: Bearer $K" "$API/instances/?owner=me" | python3 -c 'import json,sys
print(next((x.get("intended_status") or "" for x in json.load(sys.stdin)["instances"] if x["id"]==int(sys.argv[1])), "gone"))' $1; }
python3 $P/make_inputs.py $OUT/in --n $KC_N > /dev/null
# af2ig (dl_binder_design: chain A = binder) and esmc inputs, appended to the digest list box_arms checks
python3 $P/af2ig_inputs.py $P/fixtures $OUT/in/af2ig && cp $P/../../scripts/gpu_vs_tt/fixtures/prot512.seq $OUT/in/ &&
  (cd $OUT/in && for f in af2ig/* prot512.seq; do echo "$(sha256sum $f | cut -d' ' -f1)  /root/kc/in/$f"; done >> SHA256SUMS)
KITS=$(for e in $MODELS; do case $e in stock-*) ;; *) printf "%s " ${e%%:*};; esac; done)
tar -czf $OUT/kit.tgz -C "$KIT" common $KITS
# the stock rows run tt-bio's own GPU harnesses (box_stock.sh), from this checkout, under /work on the box
tar -czf $OUT/work.tgz -C $P/../.. scripts/gpu_vs_tt perf/nesso1 perf/dsfix examples/ground_truth_structures
for _ in $(seq $NMAX); do
  if [ -z "${ADOPT:-}" ]; then
    # WAIT_H: no matching offer yet (the driver >= 580 boxes are often all rented) -> look again every 10 min, bounded
    O=$(next_offer); while [ -z "$O" ] && [ $(( $(date +%s) - TW )) -lt $(python3 -c "print(int(${WAIT_H:-0}*3600))") ]; do
      sleep 600; O=$(next_offer); done
    [ -z "$O" ] && { log "no offer matches: $Q"; break; }; TRIED="$TRIED$O "
    C=$(credit); python3 -c "import sys; sys.exit(0 if $C - $CAP_H * ${MAXDPH:-1.5} >= $FLOOR else 1)" ||
      { log "REFUSED at offer $O: credit \$$C - $CAP_H h x \$${MAXDPH:-1.5}/h would cross the \$$FLOOR floor"; exit 4; }
    echo "-  $CAP_H  $SLUG  # $L creating on offer $O" >> $B
    I=$($V create instance $O --image nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 --disk ${DISK:-250} --ssh --direct --label $SLUG-$L --raw 2>&1 | grep -oP "new_contract['\"]:\s*\K[0-9]+")
    if [ -z "$I" ]; then sed -i '$d' $B; log "offer $O: create failed"; continue; fi
    mkdir -p $HOLD; echo $$ > $HOLD/$I; sed -i "\$s|.*|$I  $CAP_H  $SLUG|" $B; echo "# ^ $SLUG $L offer $O created $(date -u +%FT%TZ) by perf/kitcmp/session.sh" >> $B
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
    # MAXW: one operating point per GPU (A100: the 400 W SXM4 80 GB, not the 500 W HGX variant)
    [ "${W:-0}" -le "$MAXW" ] || { destroy $I "power limit ${W:-?} W > $MAXW"; continue; }
    BW=$(timeout 40 $S 'curl -s -o /dev/null -m 20 -w "%{speed_download}" https://download.pytorch.org/whl/cu130/torch-2.9.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl' 2>/dev/null | tail -1 | cut -d. -f1)
    log "$I download ${BW:-?} B/s"
    [ "${BW:-0}" -ge 20000000 ] || { destroy $I "download ${BW:-?} B/s < 20 MB/s"; continue; }
  else  # ADOPT=<label>: take over that session's live box (kill its session.sh first) instead of renting a new one
    read -r I O < $P/results/$ADOPT/instance; S=$(cat $P/results/$ADOPT/ssh); IP=${S##*root@}; T0=$(date +%s)
    [ "$(state $I)" = running ] || { log "ADOPT: $I is not running"; exit 5; }
    mkdir -p $HOLD; echo $$ > $HOLD/$I; sed -i "s|^$I  [0-9.]*  $SLUG$|$I  $CAP_H  $SLUG|" $B; log "adopted $I from session $ADOPT, cap $CAP_H h from now"
  fi
  echo "$S" > $OUT/ssh; echo "$I $O" > $OUT/instance
  R(){ rsync -a -e "${S% root@*}" "$@"; }; H=root@$IP
  $S 'mkdir -p /root/kit /root/kc/in /root/kc/results /work' && R $OUT/kit.tgz $OUT/work.tgz $H:/root/ && $S 'tar -xzf /root/kit.tgz -C /root/kit && tar -xzf /root/work.tgz -C /work' &&
    R $OUT/in/ $H:/root/kc/in/ && R $OUT/gpu.env $P/box_stock.sh $P/box_setup.sh $P/box_arms.sh $P/box_bc_setup.sh $P/box_bc_arms.sh $P/bc2_gpu.py $P/bc2_traj.py $P/esmc_gpu.py $P/burn_gate.py $P/kctime.py $H:/root/kc/ || { destroy $I "push failed"; continue; }
  # files whose upstream URL fails from vast, pushed from pc (sha256 checked by the kit's own install)
  for m in protenix_v2 opendde; do case " $KITS " in *" $m "*)
    W=/weights/${m%_v2}; $S "mkdir -p $W/common $W/checkpoint" && R $PKL $H:$W/common/ || { destroy $I "pkl push failed"; continue 2; };; esac; done
  case " $KITS " in *" protenix_v1 "*) $S "mkdir -p /weights/protenix_v1/common /weights/protenix_v1/checkpoint" && R $PKL $H:/weights/protenix_v1/common/ &&
    R $PXV1 $H:/weights/protenix_v1/checkpoint/protenix_base_default_v0.5.0.pt || { destroy $I "v1 push failed"; continue; };; esac
  case " $KITS " in *" protenix_v2 "*) R $PXCKPT $H:/weights/protenix/checkpoint/protenix-v2.pt || { destroy $I "ckpt push failed"; continue; };; esac
  # adopted: the new chain waits for the old one, runs PRE (shell, on the box) and reports through its own done marker
  DM=CHAIN-DONE${ADOPT:+-$L}
  R $P/box_chain.sh $H:/root/kc/ && timeout 60 $S -f "cd /root/kc && setsid nohup bash -c 'while pgrep -f \"^bash box_chain.sh\" >/dev/null; do sleep 15; done; ${PRE:-} DONE_MARK=$DM PRUNE=${PRUNE:-} exec bash box_chain.sh \"$MODELS\"' > chain-$L.log 2>&1 < /dev/null &"
  END=""
  while :; do
    sleep 300
    R $H:/root/kc/results/ $OUT/results/ 2>/dev/null; R $H:'/root/kc/*.log' $H:/root/kc/CHAIN-PROGRESS $OUT/ 2>/dev/null
    st=$(timeout 30 $S "ls /root/kc/$DM /root/kc/THERMAL-FAIL" 2>/dev/null)
    case "$st" in *THERMAL-FAIL*) END=thermal;; *$DM*) END=done;; esac
    [ -z "$st" ] && [ "$(state $I)" != running ] && END=stopped
    [ $(( $(date +%s) - T0 )) -ge $(python3 -c "print(int($CAP_H*3600-600))") ] && END=${END:-cap}
    [ -n "$END" ] && break
  done
  log "$I ended: $END ($(cat $OUT/CHAIN-PROGRESS 2>/dev/null | tr '\n' ' '))"
  R $H:/root/kc/results/ $OUT/results/ 2>/dev/null; R $H:'/root/kc/*.log' $OUT/ 2>/dev/null
  destroy $I "session end: $END"; log "credit \$$(credit) at end"
  # stopped: models this box finished stay in results/ (tagged with the box) and leave MODELS; the rest move aside and rerun
  if [ "$END" = stopped ]; then
    mkdir -p $OUT/results.stopped-$I
    for d in $OUT/results/*/; do d=${d%/}; m=${d##*/}; m=${m#setup-}
      if grep -qx "$m 0" $OUT/CHAIN-PROGRESS 2>/dev/null; then echo "$I $O" > $d/box; else mv $d $OUT/results.stopped-$I/; fi; done
    grep ' 0$' $OUT/CHAIN-PROGRESS 2>/dev/null | cut -d' ' -f1 >> $OUT/MODELS-DONE
    MODELS=$(for e in $MODELS; do grep -qx "${e%%:*}" $OUT/MODELS-DONE 2>/dev/null || printf "%s " $e; done)
    log "resuming on a new box with: $MODELS"; [ -n "${MODELS// }" ] || { echo done > $OUT/SESSION-END; exit 0; }
    continue
  fi
  [ "$END" = thermal ] && { mv $OUT/burn.log $OUT/burn-hot-$I.log; rm -rf $OUT/results; continue; }
  for d in $OUT/results/*/; do [ -f $d/box ] || echo "$I $O" > $d/box; done
  echo "$END" > $OUT/SESSION-END; log "SESSION-END $END"; exit 0
done
log "SESSION-END no-offer-passed"; echo no-offer > $OUT/SESSION-END

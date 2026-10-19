#!/bin/bash
# One unattended CHEAP-GPU session for BC4's stock reference distribution, run DETACHED on pc. Rents the cheapest
# offer with enough VRAM, pushes the BindCraft 2 harness and the banked campaigns, runs perf/spd/bc4_quality_session.sh
# on the box, pulls results every 3 minutes, and destroys the box when the session ends or at the cap.
#   usage: bc4_quality_box.sh <label> <cap_h> ['<vastai search query>'] [max_attempts]
# Status on stdout (redirect it); results perf/kitcmp/results/<label>/.
#
# Why this is not bc4_session.sh with a different query:
#  * No power-limit gate. bc4_session.sh refuses a B200 outside 990-1010 W because it measures SECONDS, and a
#    throttled card makes a timing arm a lie. This session measures DESIGNS. A campaign's outcome is a function of
#    (sha, settings, seed, trajectory), so a slow card produces the same distribution for less money, and rejecting
#    a box on its power limit here would only cost rentals.
#  * The card is picked on price, not on model. The reference is the cheapest card with enough VRAM for the
#    measured 36,356 MiB peak, i.e. 48 GB, which is roughly a tenth of a B200's hourly rate.
#  * The banked campaigns go UP with the harness. They are 14 trajectories the 2026-10-19 B200 already paid for and
#    the reference pools them with whatever this session adds; the box has no route back to pc to fetch them itself.
set -u
L=${1:?usage: bc4_quality_box.sh <label> <cap_h> [query] [attempts]} CAP_H=${2:?cap hours} NMAX=${4:-6}
Q=${3:-'num_gpus=1 gpu_ram>=45 cuda_max_good>=13.0 inet_down>=200 reliability>=0.95 disk_space>=140 dph<=0.50'}
SLUG=${SLUG:-bc4-quality}
P=$(cd "$(dirname "$0")" && pwd); WT=$(cd "$P/../.." && pwd); V=~/.vast-venv/bin/vastai
B=/home/moritz/.coworker/state/vast-budget; HOLD=/home/moritz/.coworker/state/vast-holder
S4=/home/moritz/.coworker/state/bc4; BANKED=/home/moritz/.coworker/artifacts/bc4-quality-ref
REQ=${REQ:-$S4/QUALITY-REF-REQUEST-CHEAP.json}
K=$(cat ~/.config/vastai/vast_api_key); API=https://console.vast.ai/api/v0; FLOOR=${VAST_FLOOR_USD:-1}
# The on-box wall sits inside the billed cap: the box still has to build the reference and be destroyed after the
# last wave, and a session that runs to the billed second leaves no room for either.
BUDGET_MIN=${BUDGET_MIN:-$(python3 -c "print(int($CAP_H*60) - 12)")}
CONC=${CONC:-1} MEMFRAC=${MEMFRAC:-0.85} BC4_S_PER_TRAJ=${BC4_S_PER_TRAJ:-440}
OUT=$P/results/$L; mkdir -p "$OUT"
TRIEDF=$OUT/tried; touch "$TRIEDF"; TRIED=" $(tr '\n' ' ' < "$TRIEDF")"
log(){ echo "$(date -u +%FT%TZ) $*"; }
credit(){ curl -s -H "Authorization: Bearer $K" "$API/users/current/" | python3 -c 'import json,sys;print(json.load(sys.stdin)["credit"])'; }
gone(){ curl -s -H "Authorization: Bearer $K" "$API/instances/?owner=me" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["instances"]))'; }
state(){ curl -s -H "Authorization: Bearer $K" "$API/instances/?owner=me" | python3 -c 'import json,sys
print(next((x.get("intended_status") or "" for x in json.load(sys.stdin)["instances"] if x["id"]==int(sys.argv[1])), "gone"))' $1; }
destroy(){ curl -s -X DELETE -H "Authorization: Bearer $K" "$API/instances/$1/" >/dev/null; sleep 10
  local n; n=$(gone); log "DESTROYED $1 ($2); instances left: $n"; rm -f $HOLD/$1
  sed -i "s|^$1  $CAP_H  $SLUG$|#$1 DESTROYED $(date -u +%FT%TZ) ($2), confirmed instances=$n|" $B; }
# Emits "id dph gpu_name". The rate is carried out of the search because it is the only place it is free to
# read: once the instance is destroyed the offer is gone from the listing, and the grader then cannot price
# the campaign at all. On 2026-10-19 that cost an invoice dig, and until it was done the reference priced a
# $0.4469/h card's trajectories at a B200's $6.251/h.
next_offer(){ $V search offers "$Q" -o dph --raw 2>/dev/null | python3 -c 'import json,sys
t=sys.argv[1].split()
o=next((o for o in json.load(sys.stdin) if str(o["id"]) not in t), None)
print("%s %s %s" % (o["id"], o.get("dph_total") or 0, o.get("gpu_name") or "?") if o else "")' "$TRIED"; }

[ -f "$REQ" ] || { log "FATAL: no request at $REQ"; exit 2; }
# Every local path the push stage needs, checked BEFORE the first rental. On 2026-10-19 this list named a
# bc2_traj.py that never existed in the repo, and three healthy boxes were created, failed the rsync and were
# destroyed before the loop ran out of offers. A missing file on pc must cost a second, not a rental.
PUSH="$P/bc2_gpu.py $P/box_bc_setup.sh $WT/perf/spd/bc4_quality_session.sh $WT/perf/spd/bc4_session_plan.py $WT/perf/spd/bc2_quality.py $REQ"
for f in $PUSH; do [ -e "$f" ] || { log "FATAL: push list names a missing file: $f"; exit 2; }; done
[ -d "$BANKED" ] || { log "FATAL: no banked campaigns at $BANKED"; exit 2; }
MAXDPH=$(echo "$Q" | grep -oP 'dph<=\K[0-9.]+'); C=$(credit)
python3 -c "import sys; sys.exit(0 if $C - $CAP_H * ${MAXDPH:-0.5} >= $FLOOR else 1)" ||
  { log "REFUSED: credit \$$C - $CAP_H h x \$${MAXDPH:-0.5}/h would cross the \$$FLOOR floor"; exit 4; }
log "credit \$$C at start, cap $CAP_H h, on-box budget $BUDGET_MIN min, concurrency $CONC, request $REQ"

for _ in $(seq $NMAX); do
  OFFER=$(next_offer); [ -z "$OFFER" ] && { log "no untried offer matches: $Q"; break; }
  read -r O ODPH OGPU <<< "$OFFER"
  TRIED="$TRIED$O "; echo "$O" >> "$TRIEDF"
  C=$(credit); python3 -c "import sys; sys.exit(0 if $C - $CAP_H * ${MAXDPH:-0.5} >= $FLOOR else 1)" ||
    { log "REFUSED at offer $O: credit \$$C would cross the \$$FLOOR floor"; exit 4; }
  echo "-  $CAP_H  $SLUG  # $L creating on offer $O" >> $B
  I=$($V create instance $O --image nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 --disk ${DISK:-130} --ssh --direct --label $SLUG-$L --raw 2>&1 | grep -oP "new_contract['\"]:\s*\K[0-9]+")
  [ -z "$I" ] && { sed -i '$d' $B; log "offer $O: create failed"; continue; }
  mkdir -p $HOLD; echo $$ > $HOLD/$I; sed -i "\$s|.*|$I  $CAP_H  $SLUG|" $B
  echo "# ^ $SLUG $L offer $O created $(date -u +%FT%TZ) by perf/kitcmp/bc4_quality_box.sh" >> $B
  T0=$(date +%s); log "created $I on offer $O at \$$ODPH/h ($OGPU)"; S=""
  printf '%s %s %s\n' "$I" "$ODPH" "$OGPU" > $OUT/dph
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
  VRAM=$(timeout 30 $S 'nvidia-smi --query-gpu=memory.total,name --format=csv,noheader,nounits' 2>/dev/null | head -1)
  log "$I up at $HOST:$PORT, GPU: ${VRAM:-?}"
  # The only hardware property this session needs is that the measured 36,356 MiB peak fits.
  MIB=$(echo "$VRAM" | cut -d, -f1 | tr -dc 0-9)
  [ "${MIB:-0}" -ge 45000 ] || { destroy $I "only ${MIB:-?} MiB of VRAM, the campaign peaks at 36,356"; continue; }
  echo "$S" > $OUT/ssh; echo "$I $O" > $OUT/instance; echo "$VRAM" > $OUT/gpu

  R(){ rsync -a -e "${S% root@*}" "$@"; }; H=root@$HOST
  $S 'mkdir -p /root/kc /root/tt-bio/perf/spd /root/tt-bio/perf/kitcmp' &&
    R $P/bc2_gpu.py $P/box_bc_setup.sh $H:/root/tt-bio/perf/kitcmp/ &&
    R $WT/perf/spd/bc4_quality_session.sh $WT/perf/spd/bc4_session_plan.py $WT/perf/spd/bc2_quality.py \
      $H:/root/tt-bio/perf/spd/ &&
    R "$REQ" $H:/root/kc/quality-req.json &&
    R $BANKED/ $H:/root/kc/banked/ ||
    { destroy $I "push failed"; continue; }
  log "$I: harness and $(find $BANKED -name runs.jsonl | wc -l) banked campaigns pushed"

  timeout 60 $S -f "cd /root/kc && setsid nohup env BANKED_FROM=/root/kc/banked MEMFRAC=$MEMFRAC \
      BC4_S_PER_TRAJ=$BC4_S_PER_TRAJ WT=/root/tt-bio BC2=/root/bc2 \
      bash -c 'bash /root/tt-bio/perf/spd/bc4_quality_session.sh --budget-min $BUDGET_MIN --concurrency $CONC \
        --request /root/kc/quality-req.json --out /root/kc/bc4-quality > /root/kc/quality-session.log 2>&1; \
        echo \$? > /root/kc/bc4-quality/SESSION-RC' < /dev/null &" ||
    { destroy $I "launch failed"; continue; }
  log "$I running bc4_quality_session.sh"

  # CSV and jsonl only: the .cif structures are large, nothing in the grade reads them, and the pull runs every
  # 3 minutes so that a box the host stops still leaves every finished trajectory on pc.
  PULL(){ R --include='*/' --include='*.csv' --include='*.jsonl' --include='*.log' --include='*.json' \
             --include='rc' --include='SESSION-RC' --exclude='*' $H:/root/kc/bc4-quality/ $OUT/bc4-quality/ 2>/dev/null
          R $H:'/root/kc/*.log' $OUT/ 2>/dev/null; }
  END=""
  while :; do
    sleep 180
    PULL
    [ -f $OUT/bc4-quality/SESSION-RC ] && END=done
    [ -z "$END" ] && [ "$(state $I)" != running ] && END=stopped
    [ $(( $(date +%s) - T0 )) -ge $(python3 -c "print(int($CAP_H*3600-420))") ] && END=${END:-cap}
    [ -n "$END" ] && break
  done
  PULL
  log "$I ended: $END, $(find $OUT/bc4-quality -name runs.jsonl 2>/dev/null | wc -l) campaign dirs pulled"
  destroy $I "session end: $END"
  log "credit \$$(credit) at end"; echo "$END" > $OUT/SESSION-END; exit 0
done
log "SESSION-END no-offer-passed"; echo no-offer > $OUT/SESSION-END

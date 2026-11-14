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
# Which script runs on the box. Session 5 (D-fairgpu-06) runs box_rfd4_s5.sh: the unfinished ladders only.
BOX=${BOX:-box_rfd4.sh}
P=$(cd "$(dirname "$0")" && pwd)
S4=${S4:-/home/moritz/.coworker/state/rfd4}
OUT=$P/results/$L; mkdir -p "$OUT"
# Offers we have already paid for and lost live on disk, not in this process: a relaunch after a crash used to
# re-pick the offer it had just destroyed (2026-10-19, offer 50255941 twice, ~$2.8).
TRIEDF=$OUT/tried; touch "$TRIEDF"
# Offer choice (cheapest secure offer, machines that failed twice skipped), the ledger line with its expected $, the
# dollar-capped ssh wait, power check, destroy and strikes are the fleet's one copy: ~/.coworker/scripts/vast/lib.sh.
. "${VL:-/home/moritz/.coworker/scripts/vast}/lib.sh"; VL_BY=perf/kitcmp/rfd4_session.sh VL_ANY_HOSTING=${VL_ANY_HOSTING-1}
log(){ vl_log "$@"; }
destroy(){ vl_destroy "$@"; }

MAXDPH=$(echo "$Q" | grep -oP 'dph<=\K[0-9.]+')
vl_floor_ok ${MAXDPH:-7} || { log "REFUSED: credit \$$(vl_credit) - $CAP_H h x \$${MAXDPH:-7}/h would cross the \$${VAST_FLOOR_USD:-1} floor"; exit 4; }
log "credit \$$(vl_credit) at start, cap $CAP_H h"

for _ in $(seq $NMAX); do
  vl_pick "$L" "$Q" "$TRIEDF" "${NEED_VRAM:-0}" || { log "no untried offer matches: $Q"; break; }
  echo "$O" >> "$TRIEDF"
  vl_floor_ok $ODPH || { log "REFUSED at offer $O: credit \$$(vl_credit) would cross the floor"; exit 4; }
  vl_rent nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 ${DISK:-150} || { log "offer $O: create failed"; continue; }
  vl_ssh || { destroy $I "no ssh: intended_status=$(vl_state $I), see nossh-$I.log"; continue; }
  vl_power_ok $MINW $MAXW || { destroy $I "power limit ${W:-?} W outside $MINW-$MAXW"; continue; }
  echo "$S" > $OUT/ssh; echo "$I $O" > $OUT/instance
  R(){ rsync -a -e "${S% root@*}" "$@"; }; H=root@$HOST
  $S 'mkdir -p /root/kc' &&
    R $P/rfd4_gpu.py $P/rfd4_arms.py $P/$BOX $H:/root/kc/ &&
    R $S4/inputs/ $H:/root/kc/inputs/ && echo "$I $O" | timeout 30 $S 'cat > /root/kc/instance' ||
    { destroy $I "push failed"; continue; }
  # BOXENV carries the box script's knobs across ssh (CELLS, FINAL_WARM, CUDA_VISIBLE_DEVICES). An env var
  # set around this script never reaches the remote shell, so without it a CELLS list silently ran the default.
  timeout 60 $S -f "cd /root/kc && setsid nohup env ${BOXENV:-} bash $BOX $CAP_H > box_rfd4.log 2>&1 < /dev/null &" ||
    { destroy $I "launch failed"; continue; }
  log "$I running $BOX"
  END=""
  while :; do
    sleep 180
    R $H:/root/kc/rfd4/ $OUT/rfd4/ 2>/dev/null; R $H:'/root/kc/*.log' $OUT/ 2>/dev/null
    st=$(timeout 30 $S 'ls /root/kc/rfd4/RFD4-DONE' 2>/dev/null)
    [ -n "$st" ] && END=done
    # box_rfd4.sh writes RFD4-FAILED when the import gate trips or no arm produced a warm batch. Name it, so a
    # session that measured nothing never reads as "done" in this log (session 1, 2026-10-21).
    f=$(timeout 30 $S 'cat /root/kc/rfd4/RFD4-FAILED 2>/dev/null' 2>/dev/null)
    [ -n "$f" ] && END="FAILED ($f)"
    [ -z "$st" ] && [ "$(vl_state $I)" != running ] && END=stopped
    [ $(( $(date +%s) - T0 )) -ge $(python3 -c "print(int($CAP_H*3600-420))") ] && END=${END:-cap}
    [ -n "$END" ] && break
  done
  R $H:/root/kc/rfd4/ $OUT/rfd4/ 2>/dev/null; R $H:'/root/kc/*.log' $OUT/ 2>/dev/null
  log "$I ended: $END"
  destroy $I "session end: $END"
  log "credit \$$(vl_credit) at end"; echo "$END" > $OUT/SESSION-END; exit 0
done
log "SESSION-END no-offer-passed"; echo no-offer > $OUT/SESSION-END

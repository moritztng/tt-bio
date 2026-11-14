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
# Which script runs on the box. INPUTS=<dir> pushes an already-frozen inputs tree (state/mol1/inputs), so the box
# reuses those MSAs instead of searching again. QB2=user@host:dir moves bulk data box<->qb2 directly (pc's link
# cannot carry it): qb2's key is authorized on the box, qb2 pushes dir/inputs to /root/kc/qinputs, and every pull
# also copies the box's /root/kc/mol1/ref (with CIFs) to qb2 dir/<label>/ref.
BOX=${BOX:-box_mol1.sh}
P=$(cd "$(dirname "$0")" && pwd)
S4=${S4:-/home/moritz/.coworker/state/mol1}
OUT=$P/results/$L; mkdir -p "$OUT"
# Offers we have already paid for and lost live on disk, not in this process: a relaunch after a crash used to
# re-pick the offer it had just destroyed (2026-10-19, offer 50255941 twice, ~$2.8).
TRIEDF=$OUT/tried; touch "$TRIEDF"
# Offer choice (cheapest secure offer, machines that failed twice skipped), the ledger line with its expected $, the
# dollar-capped ssh wait, power check, destroy and strikes are the fleet's one copy: ~/.coworker/scripts/vast/lib.sh.
. "${VL:-/home/moritz/.coworker/scripts/vast}/lib.sh"; VL_BY=perf/kitcmp/mol1_session.sh VL_ANY_HOSTING=${VL_ANY_HOSTING-1}
log(){ vl_log "$@"; }
destroy(){ vl_destroy "$@"; }

# qb2 -> box: vast's ssh proxy authenticates against keys ATTACHED to the instance, so a line appended to the box's
# authorized_keys alone was refused (session mol1-b200-ref1, 2026-11-01: gate passed, inputs never arrived, $5 spent).
# Attach through the API, append as well for the direct port, and prove qb2 gets in before the box is launched.
# The push itself retries: a single rsync died on a broken pipe and left mol1-fp32-r2 waiting 30 min for nothing (11-03).
qb2_key(){ local k; k=$($Q2 'cat ~/.ssh/id_ed25519.pub') || return 1
  $V attach ssh $I "$k" >/dev/null 2>&1; echo "$k" | timeout 30 $S 'cat >> /root/.ssh/authorized_keys' || return 1
  for _ in 1 2 3 4 5 6; do timeout 60 $Q2 "${S% root@*} $H true" >/dev/null 2>&1 && return 0; sleep 30; done
  log "$I: qb2 cannot ssh to the box"; return 1; }
R(){ timeout 600 rsync -a --exclude '*.cif' --exclude 'ref/*/' -e "${S% root@*}" "$@"; }
# A fresh box's sshd can drop the connection right after the first login answered (mol1-b200-ref9, 11-01: the push
# died on "Connection reset" 40 s after boot and the only B200 offer went on the tried list), so the push retries.
push(){ local n; for n in 1 2 3 4; do R "$@" && return 0; sleep 20; done; return 1; }
# Pull every 3 minutes until MOL1-DONE/FAILED/stopped/cap, then destroy. CIFs stay on the box: the speed arms'
# structures are not evidence, and on 2026-10-31 pc's ~25 KB/s link hung one pull on 279 MB of them for 35 minutes,
# which froze this loop (no DONE check, no cap) while the box kept billing.
QB2=${QB2:-}; Q2="ssh -o BatchMode=yes -o ConnectTimeout=15 ${QB2%%:*}"
offload(){ [ -z "$QB2" ] && return 0
  timeout 1800 $Q2 "mkdir -p ${QB2#*:}/$L && rsync -a -e '${S% root@*}' $H:/root/kc/mol1/ref ${QB2#*:}/$L/" 2>/dev/null; }
watch(){
END=""
while :; do
  sleep 180
  R $H:/root/kc/mol1/ $OUT/mol1/ 2>/dev/null; R $H:'/root/kc/*.log' $OUT/ 2>/dev/null; offload
  st=$(timeout 30 $S 'ls /root/kc/mol1/MOL1-DONE' 2>/dev/null)
  [ -n "$st" ] && END=done
  # box_mol1.sh writes MOL1-FAILED when the gate trips or no arm produced a warm batch. Name it, so a
  # session that measured nothing never reads as "done" in this log (session 1, 2026-10-21).
  f=$(timeout 30 $S 'cat /root/kc/mol1/MOL1-FAILED 2>/dev/null' 2>/dev/null)
  [ -n "$f" ] && END="FAILED ($f)"
  [ -z "$st" ] && [ "$(vl_state $I)" != running ] && END=stopped
  [ $(( $(date +%s) - T0 )) -ge $(python3 -c "print(int($CAP_H*3600-420))") ] && END=${END:-cap}
  [ -n "$END" ] && break
done
R $H:/root/kc/mol1/ $OUT/mol1/ 2>/dev/null; R $H:'/root/kc/*.log' $OUT/ 2>/dev/null; offload
log "$I ended: $END"
destroy $I "session end: $END"
log "credit \$$(vl_credit) at end"; echo "$END" > $OUT/SESSION-END; exit 0
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
  $S 'mkdir -p /root/kc' &&
    push $P/mol1_gpu.py $P/mol1_arms.py $P/mol1_inputs.py $P/mol1_ref.py $P/$BOX $H:/root/kc/ &&
    { [ -z "${INPUTS:-}" ] || R "$INPUTS/" $H:/root/kc/inputs/; } &&
    echo "$I $O" | timeout 30 $S 'cat > /root/kc/instance' &&
    { [ -z "$QB2" ] || { qb2_key && { $Q2 "ok=; for n in 1 2 3 4 5; do rsync -a --partial -e '${S% root@*}' ${QB2#*:}/inputs/ $H:/root/kc/qinputs/ && { ok=1; break; }; sleep 30; done; [ -n \"\$ok\" ] && ${S% root@*} $H touch /root/kc/QINPUTS-DONE" \
        > $OUT/qinputs.log 2>&1 & }; }; } ||
    { destroy $I "push failed"; continue; }
  # BOXENV carries the box script's knobs across ssh (CELLS, FINAL_WARM, CUDA_VISIBLE_DEVICES). An env var
  # set around this script never reaches the remote shell, so without it a CELLS list silently ran the default.
  timeout 60 $S -f "cd /root/kc && setsid nohup env MOL1_SESSION=$L ${BOXENV:-} bash $BOX $CAP_H > box_mol1.log 2>&1 < /dev/null &" ||
    { destroy $I "launch failed"; continue; }
  log "$I running $BOX"
  watch
done
log "SESSION-END no-offer-passed"; echo no-offer > $OUT/SESSION-END

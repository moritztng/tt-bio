#!/usr/bin/env bash
# One unattended rented-CPU session that docks the 428 PoseBusters complexes for dfd-eval.
#
#   usage: eval_box_session.sh <label> <cap_h> [workers] [threads] ['<vastai query>']
#   SHARD=i/n docks only that slice of the set, so N boxes can run one set in parallel.
#   run it DETACHED from pc, redirect stdout; poses land in $OUT/poses and are pulled every 5 min.
#
# The job is pure CPU torch (the host half of the grade), so the offer is chosen on cores, not on
# the card, and the GPU idles for the whole session: VL_IDLE_OK keeps vast_guard's idle rail off it.
# pc cannot do this itself. Measured on 2026-11-26: 2 workers under the fleet's normal load managed
# 4 complexes in 2785 s, which is 60 h for the set, and that load also clamps the AICLK other rows
# are measuring at.
#
# Rental mechanics (offer choice, ledger, dollar-capped ssh wait, destroy, strikes) are the fleet's
# one copy in ~/.coworker/scripts/vast/lib.sh. This script adds the dock and the pull-back.
set -u
L=${1:?usage: eval_box_session.sh <label> <cap_h> [workers] [threads] [query]} CAP_H=${2:?cap hours}
WORKERS=${3:-16} THREADS=${4:-4}
Q=${5:-'cpu_cores_effective>=48 inet_down>=200 reliability>=0.95 dph<=0.9'}
SLUG=${SLUG:-dfd-eval} S4=${S4:-/home/moritz/.coworker/state/dfd}
P=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HOME/scratch/dfd/eval}; mkdir -p "$OUT/poses"
TRIEDF=$OUT/tried-$L; touch "$TRIEDF"
. "${VL:-/home/moritz/.coworker/scripts/vast}/lib.sh"
VL_BY=scripts/diffdock_port/eval_box_session.sh VL_IDLE_OK=${VL_IDLE_OK:-1200}
log(){ vl_log "$@"; }

MAXDPH=$(echo "$Q" | grep -oP 'dph<=\K[0-9.]+')
vl_floor_ok "${MAXDPH:-0.9}" || { log "REFUSED: credit \$$(vl_credit) - $CAP_H h x \$${MAXDPH:-0.9}/h crosses the floor"; exit 4; }
log "credit \$$(vl_credit) at start, cap $CAP_H h, $WORKERS workers x $THREADS threads"

for _ in $(seq "${NMAX:-6}"); do
  vl_pick "$L" "$Q" "$TRIEDF" 0 || { log "no untried offer matches: $Q"; break; }
  echo "$O" >> "$TRIEDF"
  vl_floor_ok "$ODPH" || { log "REFUSED at offer $O: credit \$$(vl_credit) would cross the floor"; exit 4; }
  vl_rent "${IMAGE:-nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04}" "${DISK:-60}" || { echo "$OM" >> "$TRIEDF"; log "offer $O: create failed"; continue; }
  vl_ssh || { vl_destroy "$I" "no ssh: intended_status=$(vl_state "$I")"; continue; }
  echo "$S" > "$OUT/ssh-$L"; echo "$I $O" > "$OUT/instance-$L"

  SSH_CMD="$S" SHARD="${SHARD:-0/1}" bash "$P/eval_box_setup.sh" - "$WORKERS" "$THREADS" || { vl_destroy "$I" "setup failed"; continue; }
  log "$I docking: $(timeout 30 $S 'nproc') cores, \$$ODPH/h"

  bash "$P/eval_box_watch.sh" "$L" "$CAP_H" $(( T0 + $(python3 -c "print(int($CAP_H*3600))") ))
  break
done
log "session $L over"

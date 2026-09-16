#!/usr/bin/env bash
# Attribute openbind's 1536-token capacity stall to the lever or to the model.
#
# openbind's residency fold stalled at `trunk 0/4` on the default-ON tree: 05:39:25 to 05:55, no
# progress event, the compute process at 241 % CPU throughout and then killed by capacity_gate's
# own 900 s stall detector. 1536 tokens is above the 1024-token cap, so this is the one regime
# where TT_BIO_SDPA_FUSED_LARGE_S actually changes which kernel runs. It is therefore a candidate
# NO-GO and cannot be waved off.
#
# The control is the SAME TREE with the flag forced to 0, and the arms are separate PROCESSES.
# Never flip a call-time flag inside one device context: an in-process switch was what the earlier
# pass wrongly blamed a 768 aa hang on, and single-arm processes hung too.
#
# Off runs first: if the off arm stalls as well, the stall belongs to openbind at 1536 and the
# lever is cleared without paying for a second on-arm.
set -u
WT=/home/ttuser/.coworker/wt/ttx-a3-sdpa-ship-remerge
cd "$WT" || exit 1
OUT="$WT/perf/ttx_a3/gate2"
PROG="$OUT/attr_openbind_progress"
CARD=0
P=/home/ttuser/tt-bio-dev/env/bin/python3
mkdir -p "$OUT"; touch "$PROG"

export PYTHONPATH="$WT"
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3
export OF3_CKPT=/home/ttuser/.boltz/of3-p2-155k.pt
export ESM_ROOT=/home/ttuser/esm
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:ttx-a3-sdpa-ship-remerge

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$PROG"; }

run() {  # $1 = arm name, $2 = on|off
  local name="$1" arm="$2"
  grep -q " $name rc=" "$PROG" && { echo "skip $name"; return 0; }
  log "$name START arm=$arm loadavg=$(cut -d' ' -f1 /proc/loadavg)"
  local ev=()
  [ "$arm" = off ] && ev=(TT_BIO_SDPA_FUSED_LARGE_S=0)
  env "${ev[@]}" TT_VISIBLE_DEVICES=$CARD timeout 3600 \
    "$P" scripts/capacity_gate.py --models openbind --workers "tt-quietbox2:$CARD" \
    --no-card-reset --work-dir "$OUT/attr-openbind-$name" \
    --report "$OUT/capacity_openbind_$name.json" > "$OUT/attr-openbind-$name.log" 2>&1
  log "$name rc=$? loadavg=$(cut -d' ' -f1 /proc/loadavg)"
}

run off1 off
run on1  on
run off2 off
log "ATTR_DONE"

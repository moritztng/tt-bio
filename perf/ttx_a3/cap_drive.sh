#!/usr/bin/env bash
# Capacity at 1536 tokens, which is the only gate arm that folds in this lever's own regime
# (the route is gated strictly above 1024). Ordered so the highest-value result lands first:
# boltz2 on, then boltz2 off as its paired control, then the full roster.
#
# boltz2 first and alone because it is the model the 1.1856x was measured on and it runs its
# trunk at 4 heads, which is the head count with the full 36-of-50 fused reach. The paired
# control also closes the screen tier's 4-vs-12 finding the same way pytest and ux were closed.
#
# qb2 has hard-reset four times in two hours, so every step records an rc and is skipped on a
# rerun. --no-card-reset is not optional: a reset takes the whole board pair down and three
# sibling campaigns are in flight on the other cards.
set -u
WT=/home/ttuser/.coworker/wt/ttx-a3-fused-sdpa-default-ship
cd "$WT" || exit 1
OUT="$WT/perf/ttx_a3/gate"
PROG="$OUT/cap_progress"
CARD=3
P=/home/ttuser/tt-bio-dev/env/bin/python3
mkdir -p "$OUT"; touch "$PROG"

export PYTHONPATH="$WT"
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3
export OF3_CKPT=/home/ttuser/.boltz/of3-p2-155k.pt
export ESM_ROOT=/home/ttuser/esm
export TT_BIO_LEASE_CARDS=1,$CARD
export TT_BIO_LEASE_HOLDER=worker:ttx-a3-fused-sdpa-default-ship

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$PROG"; }

run_cap() {  # $1 = name, $2 = arm (on|off), rest = extra argv
  local name="$1" arm="$2"; shift 2
  grep -q " $name rc=" "$PROG" && { echo "skip $name"; return 0; }
  log "$name START arm=$arm loadavg=$(cut -d' ' -f1 /proc/loadavg)"
  local envflag=()
  [ "$arm" = off ] && envflag=(TT_BIO_SDPA_FUSED_LARGE_S=0)
  env "${envflag[@]}" TT_VISIBLE_DEVICES=$CARD timeout 14400 \
    "$P" scripts/capacity_gate.py --workers "tt-quietbox2:$CARD" --no-card-reset \
    --work-dir "$OUT/cap-$name" --report "$OUT/capacity_$name.json" "$@" \
    > "$OUT/capacity_$name.log" 2>&1
  log "$name rc=$? loadavg=$(cut -d' ' -f1 /proc/loadavg)"
}

run_cap b2_on   on  --models boltz2
run_cap b2_off  off --models boltz2
run_cap all_on  on
log "CAP_DRIVER_DONE"

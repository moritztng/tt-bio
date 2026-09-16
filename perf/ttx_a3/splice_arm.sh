#!/usr/bin/env bash
# Splice one lever into one model's size-ladder record, as gate_drive.sh's run_arm does.
# CARD=<n> picks the card; the lease grant widens to 0,<n> when the card is not 0.
set -u
WT=/home/ttuser/.coworker/wt/ttx-a3-sdpa-ship-remerge2
cd "$WT" || exit 1
OUT="$WT/perf/ttx_a3/gate6"; PROG="$OUT/progress"; mkdir -p "$OUT"; touch "$PROG"
CARD="${CARD:-0}"
export PYTHONPATH="$WT"
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3
export OF3_CKPT=/home/ttuser/.boltz/of3-p2-155k.pt
export ESM_ROOT=/home/ttuser/esm
if [ "$CARD" = 0 ]; then export TT_BIO_LEASE_CARDS=0; else export TT_BIO_LEASE_CARDS="0,$CARD"; fi
export TT_BIO_LEASE_HOLDER=worker:ttx-a3-sdpa-ship-remerge2
P=/home/ttuser/tt-bio-dev/env/bin/python3
m="$1"; name="splice-$m"
grep -q " $name rc=" "$PROG" && { echo "skip $name"; exit 0; }
printf '%s %s START card=%s loadavg=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" "$CARD" "$(cut -d' ' -f1 /proc/loadavg)" >> "$PROG"
TT_VISIBLE_DEVICES=$CARD timeout -k 30 "${ARM_TIMEOUT:-3000}" \
  $P scripts/release_gate.py --model size-ladder \
     --size-ladder-record-lever SDPA_FUSED_LARGE_S --size-ladder-models "$m" \
  > "$OUT/$name.log" 2>&1
rc=$?
printf '%s %s rc=%s loadavg=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" "$rc" "$(cut -d' ' -f1 /proc/loadavg)" >> "$PROG"
echo "ARM $name rc=$rc"

#!/usr/bin/env bash
# Run one size-ladder arm exactly as gate_drive.sh's run_arm does, into GATE_OUT's progress file.
set -u
WT=/home/ttuser/.coworker/wt/ttx-a3-sdpa-ship-remerge2
cd "$WT" || exit 1
OUT="$WT/perf/ttx_a3/gate6"; PROG="$OUT/progress"; mkdir -p "$OUT"; touch "$PROG"
CARD=0
export PYTHONPATH="$WT"
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3
export OF3_CKPT=/home/ttuser/.boltz/of3-p2-155k.pt
export ESM_ROOT=/home/ttuser/esm
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:ttx-a3-sdpa-ship-remerge2
P=/home/ttuser/tt-bio-dev/env/bin/python3
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$PROG"; }
m="$1"; name="ladder-$m"
grep -q " $name rc=" "$PROG" && { echo "skip $name"; exit 0; }
log "$name START loadavg=$(cut -d' ' -f1 /proc/loadavg)"
TT_VISIBLE_DEVICES=$CARD timeout -k 30 "${ARM_TIMEOUT:-2400}" \
  $P scripts/release_gate.py --model size-ladder --size-ladder-models "$m" \
  > "$OUT/$name.log" 2>&1
rc=$?
log "$name rc=$rc loadavg=$(cut -d' ' -f1 /proc/loadavg)"
echo "ARM $name rc=$rc"
for p in $(lsof -t /dev/tenstorrent/0 2>/dev/null); do echo "LEAKED HOLDER $p"; done
exit 0

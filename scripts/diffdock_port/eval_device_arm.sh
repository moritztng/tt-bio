#!/bin/bash
# One device accuracy arm for dfd-eval's phase 2, on qb2, driven by its own booking.
#
#   eval_device_arm.sh <name> <edge-tp gemm|tile> <chip> <seconds> <ids.json>
#
# Two things this does that a plain `nohup ... &` does not, both learned the hard way by this row:
#
# 1. It ASSERTS THE SETUP BEFORE IT BOOKS. The host run lost a whole launch to a tree that
#    imported fine and then raised CheckpointMissing in every worker, and the run still printed
#    its done-marker. Here the checkpoints are resolved through the port's own loader, the
#    subset ids are checked against the PoseBusters files on THIS host, and the device network
#    is imported, all on the CPU side. A broken setup fails in 30 s instead of after a
#    multi-hour wait for a chip window.
# 2. It has NO WALL-CLOCK DEADLINE. It owns book.py window fleet-dfd-eval-<name> on <chip> and
#    re-runs `book.py add` every minute while it waits, so the start only moves earlier and a
#    lapsed booking comes back. It starts the moment `active <chip>` names it alone and the chip
#    is free. dfd-perf lost two arms to typed-in epoch deadlines the calendar moved past.
#
# The dock is resumable (`docked()` parses rank1.sdf with RDKit), so a window that ends early
# costs only the complex in flight: the next window continues the same pool.
set -uo pipefail
name=${1:?name} etp=${2:?gemm|tile} card=${3:?chip} dur=${4:?seconds} ids=${5:?ids.json}

WT=/home/ttuser/.coworker/wt/dfd-eval
OUT=/home/ttuser/scratch/dfd/eval
PY=/home/ttuser/tt-bio-dev/env/bin/python
B=/home/ttuser/spd/feed/book.py
owner=fleet-dfd-eval-$name
poses=$OUT/poses-$name
log=$OUT/$name.log

mkdir -p "$OUT" "$poses"
say() { echo "$(date -u +%FT%TZ) $name: $*" >> "$log"; }
exec 9>"$OUT/$name.armlock"; flock -n 9 || { say "another arm holds the lock, exiting"; exit 0; }

cd "$WT" || exit 1
export PYTHONPATH=$WT

# ---------------------------------------------------------------- preflight, CPU only, no device
say "preflight, edge_tp=$etp chip=$card"
PB=/home/ttuser/scratch/dfd/pb/posebusters_benchmark_set IDS=$ids $PY - <<'EOF' >> "$log" 2>&1 || { say "PREFLIGHT FAILED, nothing booked"; exit 2; }
import json, os, sys, pathlib
pb = pathlib.Path(os.environ["PB"])
ids = json.load(open(os.environ["IDS"]))
ids = ids["ids"] if isinstance(ids, dict) and "ids" in ids else ids
missing = [i for i in ids
           if not (pb / i / f"{i}_protein.pdb").exists()
           or not (pb / i / f"{i}_ligand.sdf").exists()
           or not (pb / i / f"{i}_ligands.sdf").exists()]
if missing:
    sys.exit(f"PREFLIGHT: {len(missing)} of {len(ids)} subset complexes have no files here: {missing[:5]}")
# the loader that actually raises, not a directory listing
from tt_bio.diffdock import checkpoints
for which in ("score", "confidence"):
    ckpt, params = checkpoints.resolve(which, None)
    print(f"PREFLIGHT ckpt {which}: {ckpt}")
# the device network must import without a chip open
import tt_bio.diffdock.network, tt_bio.diffdock.edge_tile  # noqa: F401
print(f"PREFLIGHT ok: {len(ids)} complexes, device network imports")
EOF

# ---------------------------------------------------------------- take the window
free() {
  ! [ -e /home/ttuser/.coworker/state/cardblock-qb2-$card ] \
    && flock -n /home/ttuser/spd_qb2_chip$card.lock true 2>/dev/null \
    && ! fuser -s /dev/tenstorrent/$card 2>/dev/null
}
giveup=$(( $(date +%s) + 86400 ))
say "armed, owner $owner, $dur s on chip $card"
last=
while :; do
  [ "$($PY $B active $card 2>/dev/null)" = "$owner" ] && free && break
  [ $(date +%s) -ge $giveup ] && { say "no window in 24 h, giving up"; $PY $B rm $owner; exit 0; }
  w=$($PY $B add $owner $card $dur --note "dfd-eval phase 2 $etp accuracy arm, 85 complexes" 2>&1)
  [ "$w" != "$last" ] && { say "booking $w"; last=$w; }
  sleep 60
done

exec 8>/home/ttuser/spd_qb2_chip$card.lock
flock -w 900 8 || { say "gave up: chip $card lock"; $PY $B rm $owner; exit 0; }

# AICLK sampled DURING the arm. A perf number without its clock is not a measurement, and an
# accuracy arm states it anyway so a later reader can tell a clamped run from a clean one.
( while :; do /home/ttuser/.local/bin/tt-smi -s 2>/dev/null \
    | $PY -c "import json,sys;print(json.load(sys.stdin)['device_info'][$card]['telemetry'].get('aiclk'))" \
    >> "$OUT/$name.aiclk"; sleep 5; done ) &
clockpid=$!

export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:dfd-eval
say "start chip $card, poses -> $poses"
start=$(date +%s)
timeout $(( dur - 300 )) $PY scripts/diffdock_port/pose_eval.py dock \
  --ids "$ids" --out "$poses" --device tt --edge-tp "$etp" \
  --workers 1 --chunk 1 --seed 42 >> "$log" 2>&1
rc=$?
kill $clockpid 2>/dev/null
n=$(ls "$poses" 2>/dev/null | wc -l)
say "rc=$rc wall_s=$(( $(date +%s) - start )) docked_dirs=$n"
say "aiclk during: $(sort -n "$OUT/$name.aiclk" 2>/dev/null | uniq -c | tr '\n' ' ')"
[ $rc -eq 124 ] && /home/ttuser/.local/bin/tt-smi -r $card >> "$log" 2>&1
$PY $B rm $owner
say "window released"

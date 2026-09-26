#!/bin/bash
# bcx-bwbytes, in the environment bcx-seam, bcx-round, bcx-extramsa and bcx-tmplseam used.
#   CARD=<n> launch.sh <out-name> <script.py> <script args...>
#
# CARD is the UMD logical id, which TT_VISIBLE_DEVICES counts in PCI bus order. On qb1 that is NOT
# the /dev/tenstorrent/N node number; `perf/bcx_stack/stack.py:sysfs_node` resolves the sysfs path
# in the same ordering, so the AICLK is read off the card that ran.
#
# LEASE PROTOCOL, per the ask-11511 decision (standing permission, first card that frees):
# write the lease with holder=worker:bcx-bwbytes and this pid, `fuser` the node BEFORE opening,
# release on exit. The fuser check is not belt-and-braces: a lease file can read free while a
# holder cycles opens between stages, and the reverse -- a live holder with no lease -- put a card
# one human-speed catch away from a collision on this fleet at 03:00 on 2026-09-26.
set -u
: "${CARD:?set CARD to the card being taken, e.g. CARD=1 -- this row must not guess one}"
WT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$WT" || exit 1
NAME=$1; SCRIPT=$2; shift 2
OUT=$WT/perf/bcx_bwbytes/runs/$NAME
mkdir -p "$OUT"

HOST=$(hostname)
LEASES=$HOME/.coworker/state/leases
LEASE=$LEASES/$HOST-card$CARD.json
NODE=$(python3 - "$CARD" <<'PY'
import os, sys
root = "/sys/class/tenstorrent"
nodes = sorted(os.listdir(root),
               key=lambda n: os.path.basename(os.path.realpath(f"{root}/{n}/device")))
print(nodes[int(sys.argv[1])].split("!")[-1])
PY
) || { echo "cannot resolve the device node for CARD=$CARD" >&2; exit 1; }

if [ -s "$LEASE" ]; then
  echo "REFUSING: $LEASE exists -- card $CARD is leased. Read it before taking the card." >&2
  cat "$LEASE" >&2; exit 1
fi
if fuser "/dev/tenstorrent/$NODE" >/dev/null 2>&1; then
  echo "REFUSING: /dev/tenstorrent/$NODE (CARD=$CARD) has a live holder:" >&2
  fuser -v "/dev/tenstorrent/$NODE" >&2; exit 1
fi

mkdir -p "$LEASES"
printf '{"host": "%s", "card": "%s", "holder": "worker:bcx-bwbytes", "pid": %d, "node": %s, "acquired": %s, "released": null}\n' \
  "$HOST" "$CARD" "$$" "$NODE" "$(date +%s)" > "$LEASE"
trap 'rm -f "$LEASE"' EXIT INT TERM
echo "leased card $CARD (node $NODE) on $HOST as pid $$" >&2

export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bcx-bwbytes
export OMP_NUM_THREADS=${OMP:-6} MKL_NUM_THREADS=${OMP:-6} PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
exec /home/ttuser/bcx_e2e_venv/bin/python -X faulthandler "$WT/$SCRIPT" --out "$OUT" "$@"

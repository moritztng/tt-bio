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

CONFLICT=$(python3 "$WT/perf/bcx_bwbytes/lease_scan.py" "$LEASES" "$CARD" "$HOST") \
  || { echo "lease scan failed -- not taking a card on a failed check" >&2; exit 1; }
if [ -n "$CONFLICT" ]; then
  echo "REFUSING: card $CARD on $HOST is leased. Read these before taking it:" >&2
  echo "$CONFLICT" >&2; exit 1
fi
if fuser "/dev/tenstorrent/$NODE" >/dev/null 2>&1; then
  echo "REFUSING: /dev/tenstorrent/$NODE (CARD=$CARD) has a live holder:" >&2
  fuser -v "/dev/tenstorrent/$NODE" >&2; exit 1
fi

mkdir -p "$LEASES"
printf '{"host": "%s", "card": "%s", "holder": "worker:bcx-bwbytes", "pid": %d, "node": %s, "acquired": %s, "released": null}\n' \
  "$HOST" "$CARD" "$$" "$NODE" "$(date +%s)" > "$LEASE"
# The trap IS the lease protocol, and `exec` below used to replace this shell and take the trap
# with it, so the lease file outlived every run and the next reader saw a card leased to a dead
# pid. Verified rather than argued: a two-line repro ending in `exec /bin/true` leaks the file and
# the same script without `exec` does not (tests/test_launch_lease_release.py runs both). So this
# script keeps its own shell and runs python as a CHILD.
CHILD=
cleanup() {
  if [ -n "$CHILD" ] && kill -0 "$CHILD" 2>/dev/null; then
    # SIGTERM, never SIGKILL: a killed device holder leaves the card unopenable and the next open
    # hard-hangs the host, so the child is asked to close its device and given a minute to do it.
    kill -TERM "$CHILD" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$CHILD" 2>/dev/null || break; sleep 1; done
  fi
  rm -f "$LEASE"
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
echo "leased card $CARD (node $NODE) on $HOST as pid $$" >&2

export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bcx-bwbytes
export OMP_NUM_THREADS=${OMP:-6} MKL_NUM_THREADS=${OMP:-6} PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
/home/ttuser/bcx_e2e_venv/bin/python -X faulthandler "$WT/$SCRIPT" --out "$OUT" "$@" &
CHILD=$!
wait "$CHILD"; rc=$?
CHILD=
exit $rc

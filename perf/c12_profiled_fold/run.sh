#!/usr/bin/env bash
# One bounded session: hold the clock, run one phase, release. Board 410D (cards 2+3) as a pair,
# because a tt-metal SOURCE build refuses a single P300 chip -- exposing 1 of a board's 2 chips
# makes the cluster type CUSTOM and it then demands a mesh graph descriptor
# (tt_cluster.cpp:198,273; verified on this host 2026-09-17). The pair IS the legal subset, and
# tt_bio's CardSetLease leases every visible card, so both chips are held under this worker's
# identity for the session rather than left for the dispatcher to hand to a sibling row.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RUN="${RUN:?set RUN=<run name>}"
PHASE="${PHASE:?set PHASE=counts|probe|unit}"
METAL="${METAL:-/home/ttuser/tt-metal-b2z}"   # empty METAL => the shipped ttnn wheel
VIS="${VIS:-3,2}"
LEASE="${LEASE:-2,3}"
CLKNODES="${CLKNODES:-2,3}"
CLK="${CLK:-1350}"
OPSUP="${OPSUP:-0}"                 # >0 runs the phase under python -m tracy with this budget
OUTDIR="$HERE/runs/$RUN"
mkdir -p "$OUTDIR"

python3 "$HERE/clk.py" --nodes "$CLKNODES" --target "$CLK" --out "$OUTDIR/clock.jsonl" \
        > "$OUTDIR/clk.log" 2>&1 &
CLKPID=$!
trap 'kill -TERM $CLKPID 2>/dev/null; wait $CLKPID 2>/dev/null; gzip -f "$OUTDIR/clock.jsonl" 2>/dev/null' EXIT
sleep 2

if [ -n "$METAL" ]; then
  export TT_METAL_HOME="$METAL"
  export PYTHONPATH="$METAL/ttnn:$METAL/tools:$METAL"
fi
unset LD_LIBRARY_PATH
export TT_VISIBLE_DEVICES="$VIS"
export TT_BIO_LEASE_CARDS="$LEASE"
export TT_BIO_LEASE_HOLDER=worker:c12-profiled-fold
PY=/home/ttuser/tt-bio-dev/env/bin/python3

cd "$ROOT"
if [ "$OPSUP" -gt 0 ]; then
  "$PY" -m tracy -r -o "$OUTDIR/tracy" --op-support-count "$OPSUP" \
        --disable-device-data-push-to-tracy \
        -- "$HERE/unit_prof.py" --out "$OUTDIR/$PHASE.json" --phase "$PHASE" "$@"
else
  "$PY" "$HERE/unit_prof.py" --out "$OUTDIR/$PHASE.json" --phase "$PHASE" "$@"
fi
RC=$?
echo "rc=$RC"
exit $RC

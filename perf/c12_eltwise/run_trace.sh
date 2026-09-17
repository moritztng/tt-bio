#!/usr/bin/env bash
# Capture the executed ttnn graph of one Boltz-2 512 aa fold, with buffer addresses.
#
# Untimed on purpose: the graph's shape and its def-use edges do not depend on the clock or on
# which card ran it, so this needs no benchlock and no AICLK pin. Pricing comes from
# c10-fold-census's measured per-key rates, not from this run.
#
# Card: pass the card as $1 (default 2). qb2's device bring-up lock is host-wide, so this will
# park in locks_lock_inode_wait behind any other tt-bio process that is mid-open on ANY card.
set -eu
CARD="${1:-2}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/tmp/relvenv090/bin/python3}"
cd "$WT"
export PYTHONPATH="$WT"
export TT_BIO_LEASE_CARDS="$CARD"
export TT_VISIBLE_DEVICES="$CARD"
export TT_BIO_LEASE_HOLDER=worker:c12-fused-eltwise-at-pin
mkdir -p perf/c12_eltwise/runs
exec "$PY" perf/c12_eltwise/trace_graph.py --model boltz2 --size 512 \
     --out "perf/c12_eltwise/runs/trace_512_c${CARD}.json"

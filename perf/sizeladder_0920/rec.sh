#!/bin/bash
# p300c size-ladder RECORD, one model, one slice. $1 = card, $2 = model, $3 = rungs (optional).
#
# PYTHONPATH IS NOT OPTIONAL. release_gate imports tt_bio from the venv unless told otherwise,
# and this venv's tt_bio is /home/ttuser/tt-bio-dev, a clone frozen at 480ae2dfe (2026-09-01).
# Without this export the gate records a census of a three-week-old package under a `commit`
# field naming today's HEAD. It prints a warning when it happens; the warning is easy to scroll
# past in a 500-line log, so the export is here instead.
#
# Two more traps, inherited from state/tt-bio-sizeladder-p300c-refresh.md rather than rediscovered:
#
#   --size-ladder-fragment. Every p300c row lives in docs/size_ladder_baseline.d/<model>.json and
#   the fragment overlay REPLACES the model entry on read, so a record without this flag writes
#   into the monolith where the fragment shadows it and the check still scores the stale row.
#
#   --size-ladder-rungs, NOT RELEASE_GATE_SIZE_RUNGS. The env var replaces SIZE_LADDER_RUNGS
#   itself, so _flush_baseline's "rungs" field comes out claiming a one-rung ladder (three
#   sightings on the 09-15 campaign, restored by hand each time). The CLI flag filters instead.
#
# {256,512,768} must land in ONE process: those are the gated exponent intervals and
# _size_ladder_carry_rungs gates the carry on commit, host and grid but NOT on load, so a carried
# endpoint contributes a number measured under different contention and the ratio belongs to no
# pass. 640/896/1024 feed lever rows only and may be sliced one rung each.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WT" || exit 1
card="$1"; model="$2"; rungs="${3:-}"
export PYTHONPATH="$WT"
export TT_VISIBLE_DEVICES="$card" TT_BIO_LEASE_CARDS="${LEASE_CARDS:-$card}"
export TT_BIO_LEASE_HOLDER=worker:sizeladder-p300c-refresh-0920
args=(--model size-ladder --size-ladder-record --size-ladder-fragment
      --size-ladder-models "$model" --keep --load-ceiling "${CEILING:-0.8}")
[ -n "$rungs" ] && args+=(--size-ladder-rungs "$rungs")
exec /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/release_gate.py "${args[@]}"

#!/bin/bash
# p300c size-ladder RECORD, one model, one slice. $1 = card, $2 = model, $3 = rungs (optional).
#
# Two traps this script exists to not step in, both inherited from
# state/tt-bio-sizeladder-p300c-refresh.md rather than rediscovered:
#
#   --size-ladder-fragment. Every p300c row lives in docs/size_ladder_baseline.d/<model>.json and
#   the fragment overlay REPLACES the model entry on read, so a record without this flag writes
#   into the monolith where the fragment shadows it and the check still scores the stale row.
#
#   --size-ladder-rungs, NOT RELEASE_GATE_SIZE_RUNGS. The env var replaces SIZE_LADDER_RUNGS
#   itself, so _flush_baseline's "rungs" field comes out claiming a one-rung ladder (three
#   sightings on the 09-15 campaign, restored by hand each time). The CLI flag FILTERS the
#   ladder instead and leaves the declared rungs alone.
#
# {256,512,768} must land in ONE process: those are the gated exponent intervals and
# _size_ladder_carry_rungs gates the carry on commit, host and grid but NOT on load, so a carried
# endpoint contributes a number measured under different contention and the ratio belongs to no
# pass. 640/896/1024 feed lever rows only and may be sliced one rung each.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
card="$1"; model="$2"; rungs="${3:-}"
export TT_VISIBLE_DEVICES="$card" TT_BIO_LEASE_CARDS="${LEASE_CARDS:-$card}"
export TT_BIO_LEASE_HOLDER=worker:sizeladder-p300c-refresh-0920
args=(--model size-ladder --size-ladder-record --size-ladder-fragment
      --size-ladder-models "$model" --keep)
[ -n "$rungs" ] && args+=(--size-ladder-rungs "$rungs")
exec /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/release_gate.py "${args[@]}"

#!/bin/bash
# p300c size-ladder CHECK (writes no baseline), one model, one slice. $1 = card, $2 = model,
# $3 = rungs (optional). Same slicing rule as rec.sh: the gated trio {256,512,768} stays inside
# one process so an exponent is never taken across two differently loaded processes.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
card="$1"; model="$2"; rungs="${3:-}"
export TT_VISIBLE_DEVICES="$card" TT_BIO_LEASE_CARDS="${LEASE_CARDS:-$card}"
export TT_BIO_LEASE_HOLDER=worker:sizeladder-p300c-refresh-0920
args=(--model size-ladder --size-ladder-models "$model" --keep)
[ -n "$rungs" ] && args+=(--size-ladder-rungs "$rungs")
exec /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/release_gate.py "${args[@]}"

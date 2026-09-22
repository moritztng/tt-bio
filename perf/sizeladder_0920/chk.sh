#!/bin/bash
# p300c size-ladder CHECK (writes no baseline), one model. $1 = card, $2 = model.
#
# There is no rungs argument on purpose. --size-ladder-rungs is refused in check mode by the gate
# itself ("a check over a subset of the ladder passes without reading the rungs where a lever most
# often goes dark"), and narrowing it through RELEASE_GATE_SIZE_RUNGS would do exactly that
# silently. A check is the whole ladder or it is not a check.
#
# PYTHONPATH: see rec.sh. Without it this scores /home/ttuser/tt-bio-dev, a clone frozen at
# 2026-09-01, against a baseline recorded from today's tree.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WT" || exit 1
card="$1"; model="$2"
export PYTHONPATH="$WT"
export TT_VISIBLE_DEVICES="$card" TT_BIO_LEASE_CARDS="${LEASE_CARDS:-$card}"
export TT_BIO_LEASE_HOLDER=worker:sizeladder-p300c-refresh-0920
exec /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/release_gate.py --model size-ladder \
  --size-ladder-models "$model" --keep --load-ceiling "${CEILING:-0.8}"

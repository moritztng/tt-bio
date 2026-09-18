#!/usr/bin/env bash
# The FULL release gate on the flipped defaults -- every model in the default set, not a spot
# check. Both levers reach shared code (Transition is the engine's swiglu block; the cond-hoist
# guard is read by RF3's token DiT too), so the models that inherit the new defaults have to be
# gated, not just boltz-2.
set -u
WT=/home/ttuser/.coworker/wt/c13-land-first
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
echo "=== GATE START $(date -Is) commit $(git rev-parse --short HEAD) ==="
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:c13-land-first \
  "$PY" scripts/release_gate.py --keep
echo "=== GATE END rc=$? $(date -Is) ==="

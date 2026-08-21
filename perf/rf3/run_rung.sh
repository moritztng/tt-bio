#!/bin/bash
# One size-ladder rung of the RF3 end-to-end parity harness on qb1 card 0.
set -u
WT=/home/ttuser/.coworker/wt/rf3-port-p2
FIX="$1"; shift
mkdir -p "$WT/perf/rf3/ladder"
cd "$WT" || exit 1
exec env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:rf3-port-p2 \
     PYTHONPATH="$WT" ESM_ROOT=/home/ttuser/esm \
     /home/ttuser/tt-bio-dev/env/bin/python3 scripts/rf3_port/parity_end_to_end.py \
     --fixture "$FIX" --out "perf/rf3/ladder/${FIX}.json" "$@"

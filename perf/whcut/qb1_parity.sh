#!/bin/bash
# §6.1: the Blackhole parity delta the assembled branch owes over the release gate already
# run on main. K3 is the only change that reaches a 13x10 grid, so the legs are the ones K3
# can touch plus the one that sees a footprint change.
#
# `boltz2-hsa-nomsa` is the decisive leg: HSA is 585 aa, which pads to 640, which is a size
# K3 moves. The other boltz2 legs are small and confirm it does not reach them. `capacity`
# folds the 1095-token target at 50 samples.
#
# Cards 2 and 3, because 0 and 1 are held by qb1_chain.sh. `--seeds` is left unset: a bare
# integer matches no fixture seed, every leg flags BLOCKED-REF-REGEN-NEEDED and the gate
# false-passes on an empty scored set (tt-bio-release-gate-env-preconditions).
set -u
TREE=${TREE:-/home/ttuser/.coworker/wt/japanfold-wh-cutover}
PY=${PY:-/home/ttuser/tt-bio/env/bin/python3}
WORKDIR=$TREE/perf/whcut/out/parity-bh
cd "$TREE" || exit 1
echo "PARITY BH START $(date -u +%FT%TZ) head $(git rev-parse HEAD)"

ESM_ROOT=/home/ttuser/tt-research/esm PYTHONPATH="$TREE" TT_METAL_LOGGER_LEVEL=FATAL \
TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
  "$PY" scripts/full_parity_gate.py --workers qb1:2,qb1:3 --fresh --workdir "$WORKDIR" \
    --leg boltz2-trpcage-nomsa --leg boltz2-prot-nomsa --leg boltz2-prot-msa \
    --leg boltz2-ubiquitin-msa --leg boltz2-hsa-nomsa --leg capacity
echo "PARITY BH EXIT $? $(date -u +%FT%TZ)"

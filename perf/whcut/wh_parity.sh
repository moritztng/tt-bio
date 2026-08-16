#!/bin/bash
# §6.2: the first full parity gate ever run on Wormhole.
#
# Queued behind the §5.1 fold chain on the same two cards rather than stacking on them:
# over-subscribing qb1 three-deep is what cost the Blackhole perf gate its boltz2-affinity
# leg this morning. Waits on the chain PID directly, not on a pgrep pattern -- a pattern
# match also catches the status-check shells that carry the pattern in their own cmdline.
#
# Preconditions, all three of them (tt-bio-release-gate-env-preconditions plus the one the
# Blackhole run added): ESM_ROOT staged, RELEASE_GATE_MSA_DIR pointed at the 374 cached a3ms,
# and scripts/fetch_parity_fixtures.sh already run in this worktree. --seeds left unset: a
# bare integer matches no fixture seed and the gate false-passes on an empty scored set.
#
# The seven openfold3 legs are OUT OF SCOPE: openfold3 is not in the live catalog and the
# checkpoint is not on this box. A 22-of-29 tally is that exclusion, not an incomplete gate.
set -u
TREE=/home/cust-team/mthuening/whbase/wt-whcut
PY=/home/cust-team/mthuening/whbase/tt-bio/env/bin/python3
CHAIN_PID=${CHAIN_PID:-773336}
cd "$TREE" || exit 1
while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 60; done
echo "WH PARITY START $(date -u +%FT%TZ) head $(git rev-parse HEAD)"
ESM_ROOT=/home/cust-team/mthuening/esm \
RELEASE_GATE_MSA_DIR=/home/cust-team/mthuening/abag_xm/msa_cache \
PYTHONPATH="$TREE" TT_METAL_LOGGER_LEVEL=FATAL TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
  "$PY" scripts/full_parity_gate.py \
    --workers UF-EV-A13-GWH02:28,UF-EV-A13-GWH02:29 \
    --fold-timeout 4800 --fresh --workdir "$TREE/perf/whcut/out/parity-wh" \
    --leg esmc-300m --leg esmc-600m --leg saprot-35m --leg saprot-650m \
    --leg esmfold2-trpcage --leg boltz2-trpcage-nomsa --leg boltz2-prot-nomsa \
    --leg boltz2-prot-msa --leg boltz2-ubiquitin-msa --leg boltz2-hsa-nomsa \
    --leg protenix-prot-msa --leg protenix-ubq-msa --leg protenix-hsa-msa \
    --leg opendde-trpcage-nomsa --leg boltzgen --leg rfd3-featurizer
echo "WH PARITY EXIT $? $(date -u +%FT%TZ)"

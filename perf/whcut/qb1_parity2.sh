#!/bin/bash
# The five boltz2 envelope legs, re-run after scripts/fetch_parity_fixtures.sh restored the
# externalized reference binaries the first invocation was missing. Waits out the capacity
# leg rather than contending with it for cards 2 and 3.
set -u
TREE=/home/ttuser/.coworker/wt/japanfold-wh-cutover
PY=/home/ttuser/tt-bio/env/bin/python3
cd "$TREE" || exit 1
while pgrep -f "scripts/full_parity_gate.py" > /dev/null; do sleep 30; done
echo "PARITY BH2 START $(date -u +%FT%TZ) head $(git rev-parse HEAD)"
ESM_ROOT=/home/ttuser/tt-research/esm PYTHONPATH="$TREE" TT_METAL_LOGGER_LEVEL=FATAL \
TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
  "$PY" scripts/full_parity_gate.py --workers qb1:2,qb1:3 --fresh \
    --workdir "$TREE/perf/whcut/out/parity-bh2" \
    --leg boltz2-trpcage-nomsa --leg boltz2-prot-nomsa --leg boltz2-prot-msa \
    --leg boltz2-ubiquitin-msa --leg boltz2-hsa-nomsa
echo "PARITY BH2 EXIT $? $(date -u +%FT%TZ)"

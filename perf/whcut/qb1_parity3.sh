#!/bin/bash
# §6.1, third attempt, with the two things the first two got wrong now fixed.
#
# 1. WORKER HOST. Attempts 1-2 passed --workers qb1:2,qb1:3. qb1s hostname is tt-quietbox
#    and "qb1" is only an ssh alias in pcs config, unresolvable on the box itself. The gate
#    classifies a worker as remote when host != socket.gethostname(), so it tried to ssh to
#    qb1 from qb1 and three legs came back ERROR "predict exited 255" in 0 s. 255 is ssh.
# 2. LEG SET. boltz2-prot-msa and boltz2-ubiquitin-msa cannot score on any host: their
#    reference dirs carry meta.json, msa.a3m and the seeds but no ref_fp32/ref_bf16, so the
#    envelope has no reference. Verified missing on BOTH qb1 and pc, i.e. a pre-existing
#    fixture-asset gap and a carry, not something this cutover caused. Dropped from the set
#    and reported rather than left to read as a failure.
#
# What remains is the three legs that have envelope refs, including boltz2-hsa-nomsa, which
# is the decisive one: HSA is 585 aa, pads to 640, and 640 is where K3 fires. capacity
# already PASSED in attempt 1 (1620 s, peak 8.72 GiB) and is not repeated.
#
# Waits on the perf re-runs PID: qb1 has four cards and the affinity/rfd3 legs reach past
# the one they are given, which is what broke them the first time.
set -u
TREE=/home/ttuser/.coworker/wt/japanfold-wh-cutover
PY=/home/ttuser/tt-bio/env/bin/python3
RERUN_PID=${RERUN_PID:-4039628}
cd "$TREE" || exit 1
while kill -0 "$RERUN_PID" 2>/dev/null; do sleep 30; done
echo "PARITY BH3 START $(date -u +%FT%TZ) head $(git rev-parse HEAD)"
ESM_ROOT=/home/ttuser/tt-research/esm PYTHONPATH="$TREE" TT_METAL_LOGGER_LEVEL=FATAL \
TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
  "$PY" scripts/full_parity_gate.py --workers tt-quietbox:2,tt-quietbox:3 --fresh \
    --workdir "$TREE/perf/whcut/out/parity-bh3" \
    --leg boltz2-trpcage-nomsa --leg boltz2-prot-nomsa --leg boltz2-hsa-nomsa
echo "PARITY BH3 EXIT $? $(date -u +%FT%TZ)"

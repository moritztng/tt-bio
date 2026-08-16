#!/usr/bin/env bash
# Unattended chain across the gap between passes: the rest of the C-in/F fold sweep, then the
# card-free release-gate legs, then the gate itself. A single maxit pass is ~50 min and the gate
# alone is ~1.5 h, so the gate only ever completes if it runs detached between passes.
#
# cwd is THIS worktree, never a parent's -- a concluded slug's worktree gets torn down by fleet
# hygiene and a live job rooted there loses its files mid-run. TT_BIO_LEASE_HOLDER stays exported
# so the dispatcher keeps seeing card 0 as this worker's; the stale-lease reclaim has a known
# blind spot for detached chains (memory detached-gate-chain-lease-blind-spot-device-collision).
# Every device leg takes benchlock in turn, so nothing here races another worker.
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-p3
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200-p3
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3

echo "=== CHAIN START $(date -u +%FT%TZ) ==="

# 1. The rest of the fold sweep. 640 is in the window, 320 is its lower edge, 298 and 256 are
#    below it and must be inert.
bash perf/esmbeat/p3_run_cinf_sweep.sh 640 320 298 256
echo "SWEEP_REST_DONE rc=$? $(date -u +%FT%TZ)"

# 2. Card-free gate legs, run AFTER the folds so their CPU load does not land on a fold's seconds.
$PY -m pytest -q --tb=short > "$WT/.gate_cinf_pytest.log" 2>&1
echo "PYTEST rc=$? $(date -u +%FT%TZ)"
tail -5 "$WT/.gate_cinf_pytest.log"

$PY scripts/packaging_smoke.py > "$WT/.gate_cinf_packaging.log" 2>&1
echo "PACKAGING rc=$? $(date -u +%FT%TZ)"
tail -3 "$WT/.gate_cinf_packaging.log"

# 3. The gate. --workers localhost:0, never qb1:0: parse_workers compares the spec against
#    socket.gethostname() and a mismatch ssh-es every device leg to an unresolvable host (21
#    silent ERROR legs in p2, the real message only in .gate_*/logs/<leg>_seed0.log).
bash perf/esmbeat/p3_run_gate.sh
echo "CHAIN_DONE rc=$? $(date -u +%FT%TZ)"

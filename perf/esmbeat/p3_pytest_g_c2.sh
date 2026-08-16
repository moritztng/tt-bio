#!/usr/bin/env bash
# The pytest re-run owed by 10.6 item 2: the chain's pytest leg ran against the pre-G tree and
# was SIGKILLed. Card 2, so it runs beside the release gate on card 0 instead of behind it.
# test_protenix_largeN::test_fold_512_no_oom stays deselected for the same reason p3_run_gate_g.sh
# deselects it; it is settled separately against origin/main.
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-p3
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200-p3
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
echo "=== PYTEST G START $(date -u +%FT%TZ) head=$(git rev-parse --short HEAD) ==="
$PY -m pytest -q --tb=short -p no:cacheprovider \
  --deselect tests/test_protenix_largeN.py::test_fold_512_no_oom
echo "PYTEST_G_RC=$? $(date -u +%FT%TZ)"

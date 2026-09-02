#!/usr/bin/env bash
# Full 19-model SPECS gate on qb2 p300c card 2. The point of this run is the coverage claim:
# zero NO BASELINE rows, which has never been true on this box. Board is reset once up front so
# the run starts from clean device state; the gate itself opens the device once per model child.
set -eu
WT=/home/ttuser/.coworker/wt/p300c-baseline-coverage-gap
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio-dev/env/lib/python3.12/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export PYTHONPATH="$WT" TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2
export TT_BIO_LEASE_HOLDER=worker:p300c-baseline-coverage-gap
cd "$WT"
echo "=== gate start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
bash "$WT/perf/qb2p300cgap/reset_card2.sh" || { echo "reset refused, aborting"; exit 3; }
BENCHLOCK_WAIT_S=21600 bash /home/ttuser/.coworker/scripts/benchlock.sh p300cgap-fullgate -- \
  /home/ttuser/tt-bio-dev/env/bin/python scripts/perf_regression.py
rc=$?
echo "=== gate end $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$rc ==="
exit $rc

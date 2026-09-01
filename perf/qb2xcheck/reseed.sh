#!/usr/bin/env bash
# reseed.sh <model> <note> [card] — one perf_regression.py --update-baseline run, benchlocked.
set -u
WT=/home/ttuser/.coworker/wt/qb2-new-hardware-baseline-crosscheck
MODEL="${1:?model}"; NOTE="${2:?note}"; CARD="${3:-0}"
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio-dev/env/lib/python3.12/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export PYTHONPATH="$WT" TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD"
export TT_BIO_LEASE_HOLDER=worker:qb2-new-hardware-baseline-crosscheck
cd "$WT"
bash /home/ttuser/.coworker/scripts/benchlock.sh "qb2xcheck-reseed-${MODEL}" -- \
  /home/ttuser/tt-bio-dev/env/bin/python scripts/perf_regression.py \
  --model "$MODEL" --update-baseline --note "$NOTE" 2>&1 | grep -E "^\[|Wrote|GATE|benchlock: .*acquired"

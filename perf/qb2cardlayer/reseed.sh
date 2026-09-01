#!/usr/bin/env bash
# reseed.sh <model> <note> [card]
# One benchlocked perf_regression.py --update-baseline run, then promote.py moves the entry from
# the machine block (where the tool always writes) down to cards.p300c.models (the cells this task
# reconciles). Same mechanism as perf/qb2xcheck/reseed.sh, one model per invocation so each cell
# keeps its own note history: a multi-model update stamps one note over the whole block.
set -eu
WT=/home/ttuser/.coworker/wt/qb2-card-layer-baseline-reseed
MODEL="${1:?model}"; NOTE="${2:?note}"; CARD="${3:-0}"
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio-dev/env/lib/python3.12/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export PYTHONPATH="$WT" TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD"
export TT_BIO_LEASE_HOLDER=worker:qb2-card-layer-baseline-reseed
cd "$WT"
bash /home/ttuser/.coworker/scripts/benchlock.sh "cardlayer-reseed-${MODEL}" -- \
  /home/ttuser/tt-bio-dev/env/bin/python scripts/perf_regression.py \
  --model "$MODEL" --update-baseline --note "$NOTE" 2>&1 | grep -E "^\[|Wrote"
python3 "$WT/perf/qb2cardlayer/promote.py" "$MODEL"

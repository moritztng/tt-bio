#!/usr/bin/env bash
# draw.sh <model> <n_draws> [card]
# One fresh perf_regression.py process per draw, benchlocked. Same mechanism as
# perf/qb2xcheck/draw.sh (qb2-new-hardware-baseline-crosscheck); only the worktree, the lease
# holder and the output dir differ. Fresh process per draw because the dominant noise component
# on this box is a whole-process offset that warmup does not remove (protenix-v1-perf-cell-reseed).
# The draw table is rebuilt from logs/ by index.sh, never appended by hand.
set -u
WT=/home/ttuser/.coworker/wt/qb2-card-layer-baseline-reseed
MODEL="${1:?model}"; N="${2:-3}"; CARD="${3:-0}"
OUT="$WT/perf/qb2cardlayer"
mkdir -p "$OUT/logs"
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio-dev/env/lib/python3.12/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export PYTHONPATH="$WT"
export TT_VISIBLE_DEVICES="$CARD"
export TT_BIO_LEASE_CARDS="$CARD"
export TT_BIO_LEASE_HOLDER=worker:qb2-card-layer-baseline-reseed
cd "$WT"
for i in $(seq 1 "$N"); do
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  log="$OUT/logs/${MODEL}-card${CARD}-${ts}.log"
  bash /home/ttuser/.coworker/scripts/benchlock.sh "cardlayer-${MODEL}-c${CARD}-d${i}" -- \
    /home/ttuser/tt-bio-dev/env/bin/python scripts/perf_regression.py --model "$MODEL" >"$log" 2>&1
  echo "draw $i/$N $MODEL card$CARD rc=$? -> $log"
done
bash "$OUT/index.sh" >/dev/null

#!/usr/bin/env bash
# draw.sh <model> <n_draws> [card]
# One fresh perf_regression.py process per draw, each under benchlock, appended to draws.log.
# Fresh process per draw is the point: protenix-v1-perf-cell-reseed showed the dominant noise
# component on this box is a whole-process offset that warmup does not remove.
set -u
WT=/home/ttuser/.coworker/wt/qb2-new-hardware-baseline-crosscheck
MODEL="${1:?model}"; N="${2:-3}"; CARD="${3:-0}"
OUT="$WT/perf/qb2xcheck"
mkdir -p "$OUT/logs"
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio-dev/env/lib/python3.12/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export PYTHONPATH="$WT"
export TT_VISIBLE_DEVICES="$CARD"
export TT_BIO_LEASE_CARDS="$CARD"
export TT_BIO_LEASE_HOLDER=worker:qb2-new-hardware-baseline-crosscheck
cd "$WT"
for i in $(seq 1 "$N"); do
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  log="$OUT/logs/${MODEL}-card${CARD}-${ts}.log"
  bash /home/ttuser/.coworker/scripts/benchlock.sh "qb2xcheck-${MODEL}-c${CARD}-d${i}" -- \
    /home/ttuser/tt-bio-dev/env/bin/python scripts/perf_regression.py --model "$MODEL" >"$log" 2>&1
  rc=$?
  val=$(grep -oE "^\[${MODEL}\] [0-9.]+ " "$log" | awk "{print \$2}" | tail -1)
  line=$(grep -E "^${MODEL} " "$log" | tail -1)
  printf "%s\tmodel=%s\tcard=%s\tdraw=%s\trc=%s\tvalue=%s\tgate=%s\n" \
    "$ts" "$MODEL" "$CARD" "$i" "$rc" "${val:-NA}" "$(echo $line)" | tee -a "$OUT/draws.log"
done

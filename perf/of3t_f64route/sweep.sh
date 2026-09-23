#!/usr/bin/env bash
# of3t-f64route: the whole card-3 sweep, back to back on ONE frozen tree.
# Census fold per model first (the reach), then the three-arm A/B per model (the safety).
set -u
W=/home/ttuser/.coworker/wt/of3t-f64route
cd "$W"
echo "TREE $(git rev-parse HEAD) at $(date -u +%FT%TZ)"
for m in openfold3 rf3 protenix-v2; do
  echo "########## census $m"
  bash perf/of3t_f64route/census_fold.sh "$m" 3
done
for m in openfold3 rf3 protenix-v2; do
  echo "########## ab $m"
  bash perf/of3t_f64route/runinf.sh "$m" 3
done
echo "SWEEP DONE $(date -u +%FT%TZ)"

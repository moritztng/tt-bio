#!/usr/bin/env bash
# Wait for the in-flight 7BJ6_TVK control arm, then grade 7F8T_FAD (the cap-binding complex).
# Sequential on purpose: each arm loads ESM-2 650M and pc is shared.
set -u
cd /home/moritz/.coworker/wt/dfd-knn32
while pgrep -f "score_net_parity.py --id 7BJ6_TVK" > /dev/null; do sleep 30; done
OUT=/home/moritz/.coworker/wt/dfd-knn32/perf/dfd-knn32/grade \
  ./perf/dfd-knn32/grade_knn32.sh 7F8T_FAD

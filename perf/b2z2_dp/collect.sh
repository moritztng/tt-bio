#!/bin/bash
# Bring the curve's artifacts back from whglx and fold them into the table. Runs on pc: the
# DONE_CHECK reads /home/moritz/.coworker/state/b2z2_dp/scaling.json, and /home/moritz does not
# exist on whglx, so the table is written here and the measurements come over by scp.
#
#   collect.sh "<the measured bottleneck sentence>"
set -eu
WT=/home/moritz/.coworker/wt/b2z2-dp-throughput-linear
R=/home/mthuening/work/wt/b2z2-dp-throughput-linear/perf/b2z2_dp
mkdir -p "$WT/perf/b2z2_dp/curve" "$WT/perf/b2z2_dp/cap4" "$WT/perf/b2z2_dp/cap2" "$WT/perf/b2z2_dp/passive"
for d in curve cap4 cap2 passive probe; do
  scp -q "whglx-admin:$R/$d/*.json" "$WT/perf/b2z2_dp/$d/" 2>/dev/null || echo "no $d artifacts yet"
done
python3 "$WT/perf/b2z2_dp/scaling.py" --dir "$WT/perf/b2z2_dp/curve" \
  --out /home/moritz/.coworker/state/b2z2_dp/scaling.json --bottleneck "${1:-}"

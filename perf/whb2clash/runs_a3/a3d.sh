#!/bin/bash
# Close the pinning caveat: repeat the two extreme cells at Blackhole shipped K3 default (ON).
# If the split still holds at the default, no part of the finding rests on a non-default switch.
set -u
WT=/home/ttuser/.coworker/wt/wh-boltz2-640aa-clash-rootcause
RUN=$WT/perf/whb2clash/runs_a3
for ACC in P20794 P54802; do
  echo "=== $ACC k3-default start $(date -u +%FT%TZ) ==="
  timeout 3000 $WT/perf/whb2clash/run_arm.sh $RUN/bhk3_$ACC \
    $WT/perf/whb2clash/fixtures/partA/$ACC.yaml 1 - 2 $RUN/msa > $RUN/bhk3_$ACC.log 2>&1
  echo "=== $ACC k3-default rc=$? $(date -u +%FT%TZ) ==="
done
echo "A3D DONE $(date -u +%FT%TZ)"

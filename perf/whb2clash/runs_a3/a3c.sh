#!/bin/bash
# n>1 on the step, and one band past it. Two more real targets padding to 640 and two padding
# to 768, on Blackhole where no chunking gate fires at any of these lengths. If the clash
# fraction keeps climbing at 768 this is a length trend, not a cliff at one threshold.
set -u
WT=/home/ttuser/.coworker/wt/wh-boltz2-640aa-clash-rootcause
RUN=$WT/perf/whb2clash/runs_a3
for ACC in P27694 P20794 P54802 Q05823; do
  echo "=== $ACC start $(date -u +%FT%TZ) ==="
  timeout 3000 $WT/perf/whb2clash/run_arm.sh $RUN/bh_$ACC \
    $WT/perf/whb2clash/fixtures/partA/$ACC.yaml 0 - 2 $RUN/msa > $RUN/bh_$ACC.log 2>&1
  echo "=== $ACC rc=$? $(date -u +%FT%TZ) ==="
done
echo "A3C DONE $(date -u +%FT%TZ)"

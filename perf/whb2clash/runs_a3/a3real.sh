#!/bin/bash
# A1d/A1e on Blackhole -- the fixture-vs-length separation, run on the arch where no chunking
# gate can fire at either size. P00352 501 aa pads to 512, P22303 614 aa pads to 640.
set -u
WT=/home/ttuser/.coworker/wt/wh-boltz2-640aa-clash-rootcause
RUN=$WT/perf/whb2clash/runs_a3
for ACC in P00352 P22303; do
  echo "=== $ACC start $(date -u +%FT%TZ) ==="
  timeout 3000 $WT/perf/whb2clash/run_arm.sh $RUN/bh_$ACC \
    $WT/perf/whb2clash/fixtures/partA/$ACC.yaml 0 - 2 $RUN/msa > $RUN/bh_$ACC.log 2>&1
  echo "=== $ACC rc=$? $(date -u +%FT%TZ) ==="
done
echo "A3REAL DONE $(date -u +%FT%TZ)"

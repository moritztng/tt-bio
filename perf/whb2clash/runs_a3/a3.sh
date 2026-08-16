#!/bin/bash
# A3 -- arch control. Same input, same production MSA, production configuration, nothing forced.
# On Blackhole --fast the chunking consumer at tenstorrent.py:3106 cannot fire, so if 640 aa
# steps into clashes here the Wormhole chunking gate is exonerated as the cause of defect #8.
set -u
WT=/home/ttuser/.coworker/wt/wh-boltz2-640aa-clash-rootcause
RUN=$WT/perf/whb2clash/runs_a3
for N in 512 640; do
  OUT=$RUN/bh_$N
  echo "=== A3 cdk2_$N start $(date -u +%FT%TZ) ==="
  timeout 2400 $WT/perf/whb2clash/run_arm.sh $OUT \
    $WT/perf/whb2clash/fixtures/partA/cdk2_$N.yaml 0 - 2 $RUN/msa > $OUT.log 2>&1
  echo "=== A3 cdk2_$N rc=$? $(date -u +%FT%TZ) ==="
done
echo "A3 DONE $(date -u +%FT%TZ)"

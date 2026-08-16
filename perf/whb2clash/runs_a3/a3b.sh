#!/bin/bash
# Chained on the previous chain PID (4159849), never on card occupancy: the 641 leg and the real-target
# leg overlapped on card 2 earlier because the second was launched on a guess.
set -u
WT=/home/ttuser/.coworker/wt/wh-boltz2-640aa-clash-rootcause; RUN=/home/ttuser/.coworker/wt/wh-boltz2-640aa-clash-rootcause/perf/whb2clash/runs_a3
while kill -0 4159849 2>/dev/null; do sleep 5; done
# cdk2_641 died with an L1 circular-buffer clash under TT_BIO_SDPA_DIV_K=0. K3 ships ON on
# Blackhole, so re-run it at the shipped default before calling that a user-facing failure.
echo "=== 641 k3-default start $(date -u +%FT%TZ) ==="
timeout 3000 $WT/perf/whb2clash/run_arm.sh $RUN/bh_641_k3   $WT/perf/whb2clash/fixtures/partA/cdk2_641.yaml 1 - 2 $RUN/msa > $RUN/bh_641_k3.log 2>&1
echo "=== 641 k3-default rc=$? $(date -u +%FT%TZ) ==="

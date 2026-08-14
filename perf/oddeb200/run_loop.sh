#!/bin/bash
# Separate the two things the first slice-L1 patch changed at once:
#   (a) where the chunk lands, DRAM -> L1   -- screen_chunkwise.json cleared this: all 52 chunks
#       bit-exact in isolation, max_abs_diff 0.0
#   (b) when it is allocated, ttnn.chunk (all 52 up front) -> a slice loop interleaved with compute
# This runs (b) alone: the slice loop, memory_config still DRAM. If it fails parity, the loop
# restructure is the cause and any fusion that changes the allocation sequence inherits the risk.
set -u
WT=/home/ttuser/.coworker/wt/opendde-beat-b200
cd $WT || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:opendde-beat-b200 PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
FIX=$WT/perf/size512/fixtures
echo "### loop-only start $(date -Is)"
TT_BIO_TRANSITION_SLICE_L1=2 /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/gpu_vs_tt/tt_baseline.py \
  --model opendde --repeat 2 --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m \
  --label "512, slice loop, DRAM (loop restructure only)" \
  --msa-dir $WT/.msa_om512_512 --out $WT/perf/oddeb200/base_looponly_512.json
echo "RC=$?"
echo "### done $(date -Is)"

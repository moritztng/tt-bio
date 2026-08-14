#!/bin/bash
# Fold-level A/B for the L1 row slice, plus the capacity leg it owes. One benchlock hold.
#   leg 1  flag OFF (control)        512 aa
#   leg 2  flag ON                   512 aa   -> must return CIF 357c67003bb738ac..., plDDT 0.75411
#   leg 3  flag ON                   640 aa   -> no CB clash / allocator refusal at the size that binds
set -u
WT=/home/ttuser/.coworker/wt/opendde-beat-b200
cd $WT || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:opendde-beat-b200 PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
FIX=$WT/perf/size512/fixtures
O=$WT/perf/oddeb200
echo "### slab start $(date -Is)"
echo "=== leg 1: 512 aa, slice L1 OFF ==="
$PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 2 \
  --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m --label "512, sliceL1 OFF" \
  --msa-dir $WT/.msa_om512_512 --out $O/base_slaboff_512.json
echo "leg1 RC=$?"; sleep 30
echo "=== leg 2: 512 aa, slice L1 ON ==="
TT_BIO_TRANSITION_SLICE_L1=1 $PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 2 \
  --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m --label "512, sliceL1 ON" \
  --msa-dir $WT/.msa_om512_512 --out $O/base_slabon_512.json
echo "leg2 RC=$?"; sleep 30
echo "=== leg 3: 640 aa capacity, slice L1 ON ==="
TT_BIO_TRANSITION_SLICE_L1=1 $PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 1 \
  --target $FIX/cdk2x2_640.yaml --msa-a3m $FIX/cdk2x2_640.a3m --label "640, sliceL1 ON" \
  --msa-dir $WT/.msa_om512_640 --out $O/base_slabon_640.json
echo "leg3 RC=$?"
echo "### done $(date -Is)"

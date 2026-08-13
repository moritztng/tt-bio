#!/bin/bash
# The uninstrumented absolute, qb2 card 0 (BH p150a), ttnn 0.68.0. Closes doc §7.2: 88.61 s and
# 87.10 s are both an in-instrument delta subtracted from a card-2 wall, never measured raw.
# Two processes, 30 s apart -- opening the device immediately after the previous fold process exits
# threw `silicon_sysmem_manager.cpp:326` in pass 2 leg 2.
set -u
WT=/home/ttuser/.coworker/wt/opendde-beat-b200
cd $WT || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:opendde-beat-b200 PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
FIX=$WT/perf/size512/fixtures
O=$WT/perf/oddeb200
echo "=== A: uninstrumented, glue OFF (= origin/main behaviour) $(date -Is) ==="
TT_BIO_RELP_SCATTER=0 TT_BIO_OPENDDE_SEAM_BF16=0 \
  $PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 2 \
      --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m \
      --label "512 aa cdk2x2, glue OFF = origin/main" \
      --msa-dir $WT/.msa_om512_512 --out $O/base_main_512.json
echo "A RC=$?"; sleep 30
echo "=== B: uninstrumented, glue ON $(date -Is) ==="
$PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 2 \
    --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m \
    --label "512 aa cdk2x2, glue ON" \
    --msa-dir $WT/.msa_om512_512 --out $O/base_glue_512b.json
echo "B RC=$?"
echo "=== done $(date -Is) ==="

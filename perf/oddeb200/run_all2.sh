#!/bin/bash
# opendde-beat-b200 execution pass 2, qb2 card 0 (BH p150a), ttnn 0.68.0. ONE benchlock hold,
# three legs, each writing its artifact before the next starts.
#   leg 1  instrumented A/B  on,on,mmdef,glue,on   -> A/A floor, byte-identical alternative, glue delta
#   leg 2  UNINSTRUMENTED predict_one wall, glue ablated by env (= origin/main behaviour)
#   leg 3  UNINSTRUMENTED predict_one wall, glue ON
set -u
WT=/home/ttuser/.coworker/wt/opendde-beat-b200
cd $WT || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:opendde-beat-b200 PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
FIX=$WT/perf/size512/fixtures
O=$WT/perf/oddeb200

echo "=== leg 1: instrumented A/B $(date -Is) ==="
$PY -u perf/other512/fold_ab_multi.py --model opendde --sizes 512 \
    --arms on,on,mmdef,glue,on --out $O/ab_reanchor_512.json
echo "leg1 RC=$?"

echo "=== leg 2: uninstrumented, glue OFF (origin/main behaviour) $(date -Is) ==="
TT_BIO_RELP_SCATTER=0 TT_BIO_OPENDDE_SEAM_BF16=0 \
  $PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 2 \
      --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m \
      --label "512 aa cdk2x2, glue OFF = origin/main" \
      --msa-dir $WT/.msa_om512_512 --out $O/base_main_512.json
echo "leg2 RC=$?"

echo "=== leg 3: uninstrumented, glue ON $(date -Is) ==="
$PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 2 \
    --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m \
    --label "512 aa cdk2x2, glue ON" \
    --msa-dir $WT/.msa_om512_512 --out $O/base_glue_512.json
echo "leg3 RC=$?"
echo "=== done $(date -Is) ==="

#!/bin/bash
# opendde-beat-b200, one benchlock acquisition, three legs. qb2 card 1 (BH p150a), ttnn 0.68.0.
# Six other tasks were queued on the lock when this was written, so everything that needs an
# exclusive box runs inside ONE hold, in falling order of what the deliverable needs:
#
#   1  instrumented fold A/B, arms on,on,mmdef,glue,on          -- the A/A floor, the byte-identical
#      alternative, and the host-glue delta, with per-arm CIF sha256 + plDDT
#   2  UNINSTRUMENTED predict_one wall of origin/main's behaviour (glue ablated by env)  -- §7.2:
#      88.61 s and 87.10 s are both `90.874 - an in-instrument delta` measured on card 2 on a tree
#      that is now 14 commits behind, and neither has ever been measured uninstrumented
#   3  the same uninstrumented wall WITH the glue -- the number that would be published
#
# Each leg writes its own JSON before the next starts, so a truncated hold still leaves evidence.
set -u
WT=/home/ttuser/.coworker/wt/opendde-beat-b200
cd $WT || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:opendde-beat-b200 PYTHONPATH=$WT
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

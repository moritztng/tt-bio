#!/bin/bash
# The diffusion trace at 512 aa -- the last screened lever that costs hours rather than days.
# qb2 card 0 (BH p150a), ttnn 0.68.0. ONE benchlock hold, three legs.
#   leg 1  instrumented, trace region reserved for EVERY arm in the process:
#          glue,traceglue,glue,traceglue -> the delta measured twice, alternating, plus per-arm
#          CIF sha256 and plDDT, which is the only thing that decides bit-identity at this size
#   leg 2  UNINSTRUMENTED wall, glue ON, trace OFF
#   leg 3  UNINSTRUMENTED wall, glue ON, trace ON   <- the number that would be published
set -u
WT=/home/ttuser/.coworker/wt/opendde-beat-b200
cd $WT || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:opendde-beat-b200 PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
FIX=$WT/perf/size512/fixtures
O=$WT/perf/oddeb200

echo "=== leg 1: instrumented glue vs traceglue $(date -Is) ==="
TT_BIO_TRACE_REGION_SIZE=1073741824 \
  $PY -u perf/other512/fold_ab_multi.py --model opendde --sizes 512 \
      --arms glue,traceglue,glue,traceglue --out $O/ab_trace_512.json
echo "leg1 RC=$?"; sleep 30

echo "=== leg 2: uninstrumented, glue ON, trace OFF $(date -Is) ==="
$PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 2 \
    --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m \
    --label "512 aa cdk2x2, glue ON, trace OFF" \
    --msa-dir $WT/.msa_om512_512 --out $O/base_notrace_512.json
echo "leg2 RC=$?"; sleep 30

echo "=== leg 3: uninstrumented, glue ON, trace ON $(date -Is) ==="
TT_BIO_BASE_TRACE=1 TT_BIO_TRACE_REGION_SIZE=1073741824 \
  $PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 2 \
      --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m \
      --label "512 aa cdk2x2, glue ON, trace ON" \
      --msa-dir $WT/.msa_om512_512 --out $O/base_trace_512.json
echo "leg3 RC=$?"
echo "=== done $(date -Is) ==="

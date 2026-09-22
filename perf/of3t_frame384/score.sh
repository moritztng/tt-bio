#!/bin/bash
# of3t-frame384 deliverable 2: the frame-matched reading, at crop 384 and at crop 64.
# CPU only, no card, no device opened. Run on qb1 (tt-quietbox), which is where the local
# float64 reference AND the floor were built -- D189 makes the floor's host part of the claim.
set -euo pipefail
D=/home/ttuser/of3t_frame384
PY=/home/ttuser/tt-bio-dev/env/bin/python
export PYTHONPATH=$D/pkg
common=(--ref-model-f64 $D/grads_f64_043.pt
        --c64-plain $D/c64_f64_plain.pt --c64-ckpt $D/c64_f64_ckpt.pt
        --c64-bf16-plain $D/c64_bf16_plain.pt --c64-bf16-ckpt $D/c64_bf16_ckpt.pt
        --c64-banked $D/ref_f64_c64.pt
        --refs-built-on "qb1 (tt-quietbox), CPU only, no card, EPYC 8124P, torch 2.8.0+cpu / python 3.10.12"
        --arm-built-on "qb2 (tt-quietbox2) card 0, p300c Blackhole -- of3t-modelboundary banked it as dev_RENORM_n384_nocaptures.pt"
        --model-ref-built-on "qb2 (tt-quietbox2), CPU, upstream 0.4.3 full-model float64 on batch_step003")

case "${1:-both}" in
  n384|both)
    $PY $D/pkg/of3t_frame384/frame384.py "${common[@]}" --crop 384 \
      --ref-f64-n384 $D/ref_f64_n384.pt --ref-bf16-n384 $D/ref_bf16auto_n384.pt \
      --ours-n384 $D/dev_RENORM_n384_nocaptures.pt \
      --ref-f64-report $D/REF_F64_N384.json \
      --model-artifact $D/MODEL_withtrunk_n384.json \
      --reproduces 2.159527121735274 \
      --reproduces-from "perf/of3t_ditmodel/TRUNK_D174.json stats.MASKON_vs_FLOAT64" \
      --out $D/FRAME_N384.json ;;& # fall through so `both` runs the control too
  c64|both)
    # the same scorer at crop 64, where of3t-apbback already published the answer. This is the
    # generalisation control: same code, same definitions, only the padded width differs.
    $PY $D/pkg/of3t_frame384/frame384.py "${common[@]}" --crop 64 \
      --ref-f64-n384 $D/c64_f64_plain.pt --ref-bf16-n384 $D/c64_bf16_plain.pt \
      --ours-n384 $D/dev_scope_RENORM_c64.pt \
      --ref-f64-report $D/C64_F64_plain.json \
      --reproduces 1.7043040667627918 \
      --reproduces-from "perf/of3t_apbback/REFAUDIT.json cross.padshape_RENORM_w64__vs__REF_MODEL_f64" \
      --out $D/C64_QB1_FRAME.json ;;
esac

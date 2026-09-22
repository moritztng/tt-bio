#!/bin/bash
# of3t-widthattr deliverable 2: where the width growth lives. CPU only, no card, no device
# opened. Run on qb1 (tt-quietbox), the host that built BOTH crop-384 and crop-64 local
# references and both floors -- D189 makes the floor's host part of every ratio.
set -euo pipefail
D=/home/ttuser/of3t_frame384
W=/home/ttuser/of3t_widthattr
PY=/home/ttuser/tt-bio-dev/env/bin/python
mkdir -p $W
export PYTHONPATH=$W/pkg
OMP_NUM_THREADS=${OMP:-16} $PY $W/pkg/of3t_widthattr/widthgrowth.py \
  --ref384 $D/ref_f64_n384.pt        --ref64  $D/c64_f64_plain.pt \
  --floor384 $D/ref_bf16auto_n384.pt --floor64 $D/c64_bf16_plain.pt \
  --ours384 $D/dev_RENORM_n384_nocaptures.pt --ours64 $D/dev_scope_RENORM_c64.pt \
  --boundary384 $D/boundary_n384.pt  --boundary64 $D/boundary_c64.pt \
  --floor-host "qb1 (tt-quietbox), CPU only, no card, EPYC 8124P, torch 2.8.0+cpu / python 3.10.12" \
  --arm-host "qb2 (tt-quietbox2) card 0, p300c Blackhole" \
  --out $W/GROWTH.json

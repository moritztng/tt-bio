#!/bin/bash
# The host-only leg of the grade: no card, no lease, no device open.
# `grade.py --skip-device` needs none -- run_round builds the predictor with trunk="jax", so the
# capture, the torch reference stacks and the bc2_jax/f64 envelope arms are all host work. The
# device arm is the only part that needs a card and it is deliberately not run here.
set -u
WT=/home/ttuser/.coworker/wt/bcx-tmplseam
cd "$WT" || exit 1
OUT=$WT/perf/bcx_tmplseam/runs/grade_host
mkdir -p "$OUT"
unset TT_VISIBLE_DEVICES TT_BIO_LEASE_CARDS TT_BIO_LEASE_HOLDER
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
exec /home/ttuser/bcx_e2e_venv/bin/python -X faulthandler "$WT/perf/bcx_tmplseam/grade.py" \
  --out "$OUT" --skip-device --envelope

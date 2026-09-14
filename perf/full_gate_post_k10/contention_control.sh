#!/usr/bin/env bash
# Why the post-K10 perf leg FAILed 5 models, when nothing in the K10 wave can plausibly
# reach 5 models at once and boltz2 -- the model those levers actually act on -- read +6.0%.
#
# Two questions, one script, no new measurement harness: scripts/perf_regression.py is driven
# per-model, unchanged.
#
#   1. LOAD SENSITIVITY. boltz2 PASSed at +6.0% at 05:54Z on a loadavg-1.40 box. Re-read it now,
#      on the same tree with the same levers, while two other 44-leg gates hold the box at
#      loadavg ~10. If it craters, these 20 aa legs are host-bound and the FAILs are the box.
#   2. LEVER ARM. Same models with the three K10 levers forced OFF, interleaved model by model
#      with the ON arm so the two arms see the same slice of a drifting load
#      (op-ab-must-interleave-arms-compile-warmup-bias). If OFF is no faster, the levers are
#      not what moved these numbers.
#
# nesso1 and pxdesign are the two cheapest of the five FAILs (3 reps of ~20 s and ~16 s).
# boltz2-affinity at 148 s/rep is too expensive to arm-test in one pass and is left to section 8.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${CTRL_OUT:-/home/ttuser/.coworker/artifacts/tt-bio-full-gate-post-k10/contention-control}"
mkdir -p "$OUT"
cd "$WT"
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
export TT_BIO_LEASE_CARDS=3
export TT_BIO_LEASE_HOLDER=worker:tt-bio-full-gate-post-k10
export TT_BIO_LEASE_TIMEOUT=1800
export TT_VISIBLE_DEVICES=3
PY=/home/ttuser/tt-bio-dev/env/bin/python3

for m in boltz2 nesso1 pxdesign; do
  for arm in on off; do
    echo "=== $m arm=$arm start $(date -Is) loadavg $(cut -d" " -f1-3 /proc/loadavg) ==="
    if [ "$arm" = off ]; then
      env TT_BIO_TRIMUL_GP_BANK_SPLIT=0 TT_BIO_TRIMUL_MASK_L1=0 TT_BIO_RESIDUAL_L1=0 \
        "$PY" scripts/perf_regression.py --model "$m" --threshold 15 \
        --note "post-K10 contention control, arm=off" > "$OUT/$m.$arm.log" 2>&1
    else
      "$PY" scripts/perf_regression.py --model "$m" --threshold 15 \
        --note "post-K10 contention control, arm=on" > "$OUT/$m.$arm.log" 2>&1
    fi
    echo "=== $m arm=$arm exit=$? $(date -Is) loadavg $(cut -d" " -f1-3 /proc/loadavg) ==="
  done
done

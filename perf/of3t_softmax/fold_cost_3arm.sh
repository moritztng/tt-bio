#!/usr/bin/env bash
# Whole-fold cost of all THREE softmax arms, interleaved A/B/C then C/B/A.
#
# Why this exists separately from fold_ab.py: the first fold-level session paid kernel
# compilation in its first arm, so a fourth arm added later ran 48.6 s against that session's
# 103.6 s and was not comparable to it. Every arm here runs against an already-warm cache, in
# both orders, so no arm is charged for the others' compile.
#
#   none       shipped: no compute_kernel_config at the five sites
#   precise    TT_BIO_SOFTMAX_PRECISE_AB=all
#   accurate   TT_BIO_ACCURATE_SOFTMAX_AB=all   (reaches the AttentionPairBias sites, NOT
#              OpenFold3's own diffusion-transformer file, which calls ttnn.softmax directly)
set -u
WT=/home/ttuser/.coworker/wt/of3t-softmax
WORK=${1:-/home/ttuser/of3t_softmax_work/cost3}
PY=$HOME/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1

run() {
  local arm=$1 rep=$2 out="$WORK/$arm-r$rep"
  rm -rf "$out"; mkdir -p "$out"
  local pre="" acc=""
  case $arm in
    none)     pre="";    acc="" ;;
    precise)  pre="all"; acc="" ;;
    accurate) pre="";    acc="all" ;;
  esac
  local t0 t1
  t0=$(date +%s.%N)
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-softmax \
  PYTHONPATH="$WT" TT_BIO_SOFTMAX_PRECISE_AB="$pre" TT_BIO_ACCURATE_SOFTMAX_AB="$acc" \
  "$PY" -m tt_bio.main predict examples/ubq.yaml --model openfold3 --out_dir "$out" \
    --seed 0 --diffusion_samples 5 --single_sequence --output_format cif --override \
    >"$out/run.log" 2>&1
  local rc=$?
  t1=$(date +%s.%N)
  local cif="$out/openfold3_results_ubq/structures/ubq.cif"
  echo "$arm rep$rep rc=$rc wall=$(echo "$t1 - $t0" | bc) sha=$( [ -f "$cif" ] && sha256sum "$cif" | cut -c1-16 || echo NONE)"
}

# warm once so rep1 is not the arm that pays for any residual compile
run none 0 >/dev/null 2>&1
for arm in none precise accurate; do run "$arm" 1; done
for arm in accurate precise none; do run "$arm" 2; done

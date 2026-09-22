#!/usr/bin/env bash
# BoltzGen's leg of the MODELS table: does the softmax config move a DESIGN, not a fold.
# Three arms on card 0 -- lever off, lever on at the same seed, and a different seed as the
# negative control that proves the digest can move at all.
set -u
WT=/home/ttuser/.coworker/wt/of3t-softmax
SPEC=$HOME/.cache/tt-bio/regression/tt_boltz_regression_v1
WORK=${1:-/home/ttuser/of3t_softmax_work/boltzgen}
PY=$HOME/tt-bio-dev/env/bin/python3
STEPS=${STEPS:-15}

for arm in off on ctl; do
  out="$WORK/$arm"
  [ -f "$out/intermediate_designs/input.cif" ] && { echo "$arm cached"; continue; }
  rm -rf "$out"; mkdir -p "$out/spec"
  cp "$SPEC/input/design_spec.yaml" "$out/spec/input.yaml"
  cp "$SPEC/input/target.cif" "$out/spec/1g13.cif"
  case $arm in
    off) AB=""    ; SEED=0 ;;
    on)  AB="all" ; SEED=0 ;;
    ctl) AB=""    ; SEED=7 ;;
  esac
  ( cd "$WT" && \
    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-softmax \
    PYTHONPATH="$WT" TT_BIO_SOFTMAX_PRECISE_AB="$AB" \
    "$PY" -m tt_bio.main design "$out/spec/input.yaml" --model boltzgen \
      --out_dir "$out" --num_designs 1 --devices 0 --seed "$SEED" \
      --config design "sampling_steps=$STEPS" ) >"$out/run.log" 2>&1
  echo "$arm rc=$? $(sha256sum "$out/intermediate_designs/input.cif" 2>/dev/null | cut -c1-16)"
done

for arm in off on ctl; do
  f="$WORK/$arm/intermediate_designs/input.cif"
  [ -f "$f" ] && echo "$arm $(sha256sum "$f" | cut -c1-16)" || echo "$arm NO-OUTPUT"
done

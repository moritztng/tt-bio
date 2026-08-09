#!/bin/bash
# W5 (arm B, L1-residency guard fix) owes a real cross-model A/B on boltz2 and BoltzGen.
# W5 only ran smoke tests there. This runs both models through the real CLI on arms L0
# (main today) and L2 (main + D6 + W5, flag off) and diffs every output byte for byte.
set -u
WT=/home/ttuser/.coworker/wt/perfwar-qb1-rebaseline-and-land
export TT_VISIBLE_DEVICES=3
export TT_BIO_LEASE_HOLDER=worker:perfwar-qb1-rebaseline-and-land
export TT_BIO_TRIMUL_OUT_FUSED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/usr/bin/python3
LOG="$WT/perf/land/out/w5_crossmodel.log"
OUTROOT=/home/ttuser/w5_xm

run () {  # arm model spec verb extra...
  local arm=$1 model=$2 spec=$3 verb=$4; shift 4
  local od="$OUTROOT/${arm}_${model}"
  [ -d "$od" ] && { echo "SKIP $arm $model (exists)" >>"$LOG"; return 0; }
  mkdir -p "$od"
  echo "=== $(date -u +%H:%M:%S) $arm $model" >>"$LOG"
  ( cd "$WT/arms/$arm" && PYTHONPATH=$PWD timeout 2400 "$PY" -m tt_bio.main "$verb" "$spec" \
      --model "$model" --seed 0 --out_dir "$od" "$@" >>"$LOG" 2>&1 )
  echo "=== $(date -u +%H:%M:%S) $arm $model rc=$?" >>"$LOG"
}

# boltz2: 117 aa, the size W5 itself measured (1.0164x), so the L1 guard is exercised.
run L0 boltz2 examples/prot.yaml predict --sampling_steps 50 --diffusion_samples 1
run L2 boltz2 examples/prot.yaml predict --sampling_steps 50 --diffusion_samples 1
# BoltzGen: the canonical binder fixture release_gate.py gates on.
run L0 boltzgen examples/binder.yaml design
run L2 boltzgen examples/binder.yaml design

{
  echo "=== $(date -u +%H:%M:%S) DIFF"
  for m in boltz2 boltzgen; do
    a="$OUTROOT/L0_$m"; b="$OUTROOT/L2_$m"
    echo "--- $m: L0 vs L2"
    diff -r --brief "$a" "$b" 2>&1 | sed 's/^/    /' | head -40
    echo "    files_L0=$(find "$a" -type f | wc -l) files_L2=$(find "$b" -type f | wc -l)"
  done
  echo "=== $(date -u +%H:%M:%S) W5 CROSSMODEL COMPLETE"
} >>"$LOG" 2>&1

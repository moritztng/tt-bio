#!/bin/bash
# Fold-level gate for the four p3 levers on the Wormhole Galaxy (GWH02, UMD 28).
#
# One tree, two arms, switched by the levers' own env gates. A two-tree A/B was rejected: it
# cannot isolate the ONE change in this delta that has no env gate -- `PairUpdateBlock.__call__`
# now calls `transition.residual(z)` instead of `_residual(z, transition(z))`. Rung 0 below is
# the control that closes that hole, and it has to pass before any other rung means anything.
#
# The wall clock here is an OOM / gross-regression check, not a lever measurement: a single-shot
# fold on a shared prod box carries 20-30 % (memory perf-gate-single-shot-legs-recurring-false-
# alarm), and the levers are worth 1-3 %. The verdict comes from the CIF sha256, which has no
# noise floor at all.
#
# Usage: ./wh_fold_ab.sh <rung>   with rung in: control 512 640 1024 all
set -u
D=/home/cust-team/mthuening/prodsync
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3
NEW=$D/src            # tt-bio at e65b66be (the four levers)
OLD=$D/src_prod       # tt-bio at a50b12e09 (what prod serves today)
MSA="--use_msa_server --msa_dir /data/msa_cache"
COMMON="--model esmfold2 --fast --seed 0 --recycling_steps 10 --sampling_steps 100 --diffusion_samples 1 --override"

# `off` reproduces the pre-p3 code path on the new tree; `ship` is the tree's own defaults.
off_env() { echo "TT_BIO_PAIR_FFN_L1_LN=0 TT_BIO_PAIR_FFN_L1_SLICE=0 TT_BIO_PAIR_FFN_FUSED_RESIDUAL=0 TT_BIO_PAIR_FFN_FILL_ASSEMBLY=0"; }

fold() {  # fold <tag> <src> <fasta> <extra-args> <env...>
  local tag=$1 src=$2 fa=$3 extra=$4; shift 4
  local t0=$SECONDS
  rm -rf "$D/out_$tag"
  env -i HOME=/home/cust-team PATH=/usr/bin:/bin \
      HF_HUB_CACHE=/home/cust-team/models TT_VISIBLE_DEVICES=28 \
      TT_BIO_LEASE_HOLDER=worker:japanfold-prod-sync-esmfold2-levers \
      TT_BIO_DRAM_PEAK=$D/dram_$tag.log PYTHONPATH=$src "$@" \
      timeout 3000 $PY -m tt_bio.main predict "$fa" $COMMON $extra \
      --out_dir "$D/out_$tag" > "$D/run_$tag.log" 2>&1
  local rc=$? cif
  cif=$(ls "$D/out_$tag"/*/structures/*.cif 2>/dev/null | head -1)
  echo "$tag exit=$rc wall=$((SECONDS - t0))s sha=$( [ -n "$cif" ] && sha256sum "$cif" | cut -c1-16 || echo NO_CIF )"
}

case "${1:-all}" in
control|all)
  # Rung 0. Prod's tree, shipped defaults, vs the new tree with all four levers off. Same sha256
  # or nothing below is interpretable, because the ungated esmfold2.py refactor would be the
  # difference and every `off` arm would silently carry it.
  fold ctl_prod  "$OLD" "$D/cdk2_512.fasta" "$MSA"
  fold ctl_newoff "$NEW" "$D/cdk2_512.fasta" "$MSA" $(off_env)
  ;;& 
512|all)
  fold m512_off  "$NEW" "$D/cdk2_512.fasta" "$MSA" $(off_env)
  fold m512_ship "$NEW" "$D/cdk2_512.fasta" "$MSA"
  ;;&
640|all)
  # 640 aa is the deepest MSA size that folds on Wormhole today; 788 and 1024 OOM pre-lever on a
  # defect owned by japanfold-esmfold2-wh-msa-cap-p2 (its section 17), not by these levers.
  fold m640_off  "$NEW" "$D/cdk2_640.fasta" "$MSA" $(off_env)
  fold m640_ship "$NEW" "$D/cdk2_640.fasta" "$MSA"
  ;;&
1024|all)
  # Single-sequence, because the MSA path cannot reach 1024 on Wormhole yet. This is still the
  # worst case for these levers: the row block's L1 residents scale with L and 1024 is the top of
  # PAIR_FFN_ROW_BLOCK_SEQ.
  fold s1024_off  "$NEW" "$D/cdk2_1024.fasta" "--single_sequence" $(off_env)
  fold s1024_ship "$NEW" "$D/cdk2_1024.fasta" "--single_sequence"
  ;;
esac

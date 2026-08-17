#!/bin/bash
# Blackhole neutrality for the sync, on qb1 card 1. Two legs, because they answer two questions.
#
# Leg 1 -- the shipped 512 aa arm still reproduces the published cell's digest and its own off-arm
# delta. site/data/perf-512aa.json carries CIF sha256 295867277b9c137f and plDDT 0.9285 for the
# `ship` arm, measured on qb2 card 0. qb1 is a 13x10 grid, so its WALL is not the page's wall and
# must not be quoted as one; the DIGEST is grid-independent and is what neutrality means here.
#
# Leg 2 -- a deep-MSA fold, because leg 1's fixture carries a 35-sequence a3m and barely enters
# the MSA encoder. Memory correctness-sweep-tiled-fixture-measures-one-input: a tiled CDK2 ladder
# already made a deep-MSA-only defect look size-general once on this model.
set -u
WT=/home/ttuser/.coworker/wt/japanfold-prod-sync-esmfold2-levers
PY=/home/ttuser/tt-bio-dev/env/bin/python3
MSADIR=/home/ttuser/esmfold2wh_msa      # copied from GWH02 by japanfold-esmfold2-wh-msa-cap-p2
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:japanfold-prod-sync-esmfold2-levers

cd "$WT" || exit 1

# Leg 1: census + digest + wall for every arm, one process, arms round-robin so host drift hits
# each equally. `off` is the pre-p3 baseline, `ship` the tree default; noE/noCinF/noG isolate.
/home/ttuser/.coworker/scripts/benchlock.sh japanfold-prod-sync-esmfold2-levers -- \
  $PY perf/esm3p4land/fold_ab.py --model esmfold2 --size 512 --rounds 2 \
      --arms off,ship,noE,noCinF,noG \
      --out perf/whprodsync/bh_fold_ab_512_qb1c1.json

# Leg 2: the deep-MSA fold, off vs ship, CIF sha256 is the verdict.
for arm in off ship; do
  if [ "$arm" = off ]; then
    E="TT_BIO_PAIR_FFN_L1_LN=0 TT_BIO_PAIR_FFN_L1_SLICE=0 TT_BIO_PAIR_FFN_FUSED_RESIDUAL=0 TT_BIO_PAIR_FFN_FILL_ASSEMBLY=0"
  else E=""; fi
  t0=$SECONDS
  rm -rf "/tmp/bhmsa_$arm"
  env $E TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:japanfold-prod-sync-esmfold2-levers \
      PYTHONPATH=$WT $PY -m tt_bio.main predict "$MSADIR/fixtures/cdk2_512.fasta" \
      --model esmfold2 --fast --use_msa_server --msa_dir "$MSADIR/cache" --seed 0 \
      --recycling_steps 10 --sampling_steps 100 --diffusion_samples 1 \
      --out_dir "/tmp/bhmsa_$arm" --override > "/tmp/bhmsa_$arm.log" 2>&1
  cif=$(ls /tmp/bhmsa_$arm/*/structures/*.cif 2>/dev/null | head -1)
  echo "bh_msa_$arm exit=$? wall=$((SECONDS-t0))s sha=$( [ -n "$cif" ] && sha256sum "$cif" | cut -c1-16 || echo NO_CIF )"
done

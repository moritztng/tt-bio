#!/bin/bash
# one OF3 fold, one arm, one seed, one card. $1=arm(ship|fix|shipforce) $2=seed $3=card
set -u
WT=/home/ttuser/.coworker/wt/of3t-foldab
ARM=$1; SEED=$2; CARD=$3
OUT=/home/ttuser/of3t_foldab_runs/fold/${ARM}_s${SEED}
mkdir -p "$OUT"
case "$ARM" in
  ship|shipc1)      unset TT_BIO_OF3_TRI_END_BIAS_FOLLOWS_PAIR ;;
  shipforce) export TT_BIO_OF3_TRI_END_BIAS_FOLLOWS_PAIR=1 ;;
  fix)       export TT_BIO_OF3_TRI_END_BIAS_FOLLOWS_PAIR=0 ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-foldab
cd "$WT"
exec /home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict examples/ubq.yaml \
  --model openfold3 --out_dir "$OUT" \
  --msa_dir /home/ttuser/of3t_pairbias_msa --msa_cache_only \
  --diffusion_samples 5 --sampling_steps 200 --seed "$SEED" \
  --output_format cif --override

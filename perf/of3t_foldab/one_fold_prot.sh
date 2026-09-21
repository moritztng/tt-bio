#!/bin/bash
# one OF3 fold of examples/prot.yaml (117 aa, shallow MSA). $1=arm $2=seed $3=card
set -u
WT=/home/ttuser/.coworker/wt/of3t-foldab
MSA=/home/ttuser/.coworker/artifacts/tt-bio-full-gate-post-k10/gate-f072ae02f/msa/openfold3__prot__msa-colabfold_200step_5sample_4cycle_fp32cpu
ARM=$1; SEED=$2; CARD=$3
OUT=/home/ttuser/of3t_foldab_runs/prot/${ARM}_s${SEED}
mkdir -p "$OUT"
case "$ARM" in
  ship)      unset TT_BIO_OF3_TRI_END_BIAS_FOLLOWS_PAIR ;;
  fix)       export TT_BIO_OF3_TRI_END_BIAS_FOLLOWS_PAIR=0 ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-foldab
cd "$WT"
exec /home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict examples/prot.yaml \
  --model openfold3 --out_dir "$OUT" \
  --msa_dir "$MSA" --msa_cache_only \
  --diffusion_samples 5 --sampling_steps 200 --seed "$SEED" \
  --output_format cif --override

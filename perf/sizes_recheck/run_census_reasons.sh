#!/usr/bin/env bash
# 15 step 4: re-run the four small-end census cells under 95033b2f so the decline REASONS
# are recorded. qb2 card 3, same card as every 13.5 cell. No benchlock: counter read only.
WT=/home/ttuser/.coworker/wt/sizes-recheck-protenix-openfold3
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=3
export TT_BIO_LEASE_HOLDER=worker:sizes-recheck-protenix-openfold3
export ESM_ROOT=/home/ttuser/esm
export OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
for N in 128 256; do
  for M in protenix-v2 openfold3; do
    echo "=== census $M $N $(date -Is)"
    timeout -k 20 1800 python3 scripts/lever_census.py \
      --tt-bio /home/ttuser/tt-bio-dev/env/bin/tt-bio --pythonpath "$WT" \
      --label ${M}-${N} --out perf/sizes_recheck/census_${M}_${N}_qb2c3_reasons.json \
      -- predict perf/size512/fixtures/cdk2x2_${N}.yaml --model $M \
         --msa_dir $WT/.msa_recheck_${N} --msa_cache_only --out_dir /tmp/censusr_${M}_${N}
    echo "=== rc=$? $M $N $(date -Is)"
  done
done
echo "=== CENSUS ALL DONE $(date -Is)"

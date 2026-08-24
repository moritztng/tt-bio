#!/usr/bin/env bash
# Sequencer over scripts/gpu_vs_tt/tt_baseline.py --ab-env. Not a benchmark; it only orders the
# runs. CARD 0 (this worker grant, 2026-08-23 pass 2). The previous copy was pinned to card 2,
# which the release-v0-6-8 worker holds, so every run died on DeviceInUseError.
set -u
WT=/home/ttuser/.coworker/wt/protenix-opendde-token-bucket-flip-measure
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=0
cd "$WT" || exit 1
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm
# The quiet-wait is short on purpose: cards 1/2/3 carry a multi-hour release campaign whose
# folds are device-bound at 0.6-4.7 %% CPU on 32 cores, so they never clear. Loadavg is the
# thing that matters for host dispatch and it is under 1. Co-tenancy is recorded per fold.
export BENCHLOCK_WAIT_S=3600 BENCHLOCK_LOAD_WAIT_S=90
mkdir -p perf/tokenbucket

run () {  # run <tag> <model> <fixture-stem> <repeat> <out>
  local tag=$1 model=$2 stem=$3 rep=$4 out=$5
  [ -f "$out" ] && { echo "=== SKIP $tag, $out already exists ==="; return 0; }
  echo "=== $(date -Is) START $tag card $CARD load $(cut -d\  -f1-3 /proc/loadavg) ==="
  bash /home/ttuser/.coworker/scripts/benchlock.sh protenix-opendde-token-bucket-flip-measure -- \
    env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
        TT_BIO_LEASE_HOLDER=worker:protenix-opendde-token-bucket-flip-measure \
        PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm \
    "$PY" -u scripts/gpu_vs_tt/tt_baseline.py --model "$model" --repeat "$rep" \
      --ab-env TT_BIO_PROTENIX_TOKEN_BUCKET --ab-values 1,0 \
      --target perf/size512/fixtures/${stem}.yaml --msa-a3m perf/size512/fixtures/${stem}.a3m \
      --label "$tag, page protocol, qb1 card $CARD, paired token-bucket A/B" \
      --msa-dir $WT/.msa_tokenbucket \
      --keep-cif perf/tokenbucket/cif_$tag --out "$out"
  echo "=== $(date -Is) END $tag rc=$? ==="
}

run od512 opendde     cdk2x2_512 5 perf/tokenbucket/od512_paired.json
run px512 protenix-v2 cdk2x2_512 5 perf/tokenbucket/px512_paired.json
run od298 opendde     cdk2x2_298 3 perf/tokenbucket/od298_paired.json
run px298 protenix-v2 cdk2x2_298 3 perf/tokenbucket/px298_paired.json
echo "=== $(date -Is) ALL DONE ==="

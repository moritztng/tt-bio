#!/bin/bash
# Does TT_BIO_TRIATT_FUSED_HIFI reach anything on a real OpenFold3 fold?
#
# It claims 1.88x more accurate and 20.2x faster at 512 aa and carries no recorded reason for
# being off, which makes it this row's biggest untriaged number. Before any of that is believed
# the question is reach: it is the PROCESS-WIDE default for `fused_hifi`, which only sites passing
# None follow, and only the fp32-softmax route consults it. `TriangleAttention.__init__` says
# "OpenFold3 is the only model here on the fp32 route".
#
# A firing question is load-insensitive, so the box being loud does not block it, and no timing
# claim is taken from these legs.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/hififire
IN=$WT/perf/land_standing/out/deepfix/cdk2_deep_832.yaml
MSA=$WT/perf/land_standing/out/deepfix/msa
CARD=3
mkdir -p "$OUT"
cd "$WT" || exit 1

run () {
  tag=$1
  v=$2
  dump=$OUT/stats_$tag.jsonl
  rm -f "$dump"
  t0=$(date +%s)
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing \
      PYTHONPATH="$WT/perf/land_standing/out/widekfire/hook" TT_STOCK_STATS_DUMP="$dump" \
      TT_BIO_TRIATT_FUSED_HIFI="$v" \
      /home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict --model openfold3 \
        --msa_dir "$MSA" --msa_cache_only --sampling_steps 20 --diffusion_samples 1 \
        --out_dir "$OUT/res_$tag" "$IN" > "$OUT/log_$tag.txt" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "LEG $tag fused_hifi=$v rc=$rc secs=$((t1 - t0))"
  grep -o '"plddt": [0-9.]*' "$OUT/res_$tag/openfold3_results_cdk2_deep_832/results.json" 2>/dev/null
}

run off 0
run on  1
echo HIFI_FIRE_DONE

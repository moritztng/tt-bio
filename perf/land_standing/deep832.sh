#!/bin/bash
# The dividing-k A/B on a CONFIDENT 832-token fixture, which is what this lever has lacked.
#
# Every earlier arm was --single_sequence and read pLDDT 0.37, near the confidence heads' floor,
# which is why they could not decide direction. The same tiled CDK2 with a deep alignment reads
# 0.882 at 298 aa. This is that fixture at 832 tokens: 3 tiled copies of the monomer, the
# monomer's own deep alignment tiled the same way, depth 512, served from an offline cache with
# --msa_cache_only so nothing reaches the network.
#
# off / on / off. The third leg is the control and must reproduce the first exactly; at fixed
# seed this pipeline has been bit-deterministic end to end at 704 and at 832 already.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/deepfix
IN=$OUT/cdk2_deep_832.yaml
CARD=3
cd "$WT" || exit 1

run () {
  tag=$1
  dk=$2
  ( while true; do
      printf '%s %s\n' "$(cat /sys/class/tenstorrent/tenstorrent!$CARD/tt_aiclk 2>/dev/null)" \
                       "$(cut -d' ' -f1 /proc/loadavg)"
      sleep 1
    done > "$OUT/clk_$tag.txt" ) &
  sampler=$!
  t0=$(date +%s.%N)
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing \
      TT_BIO_TRIATT_DIVIDING_K="$dk" \
      /home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict --model openfold3 \
        --msa_dir "$OUT/msa" --msa_cache_only \
        --sampling_steps 20 --diffusion_samples 1 \
        --out_dir "$OUT/res_$tag" "$IN" > "$OUT/log_$tag.txt" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  kill $sampler 2>/dev/null
  echo "LEG $tag dividing_k=$dk rc=$rc wall=$(echo "$t1 - $t0" | bc)s"
  cat "$OUT/res_$tag/openfold3_results_cdk2_deep_832/results.json" 2>/dev/null | tr -d '\n '
  echo
  sha256sum "$OUT/res_$tag/openfold3_results_cdk2_deep_832/"*.cif 2>/dev/null | head -2
}

run d832_off1 0
run d832_on   1
run d832_off2 0
echo AB_DONE

#!/bin/bash
# Score TT_BIO_TRIATT_FUSED_HIFI against a DEPOSITED structure, which is the one instrument its
# verdict has lacked.
#
# The gate could not do it: at 117 aa the lever serves 0 of 432 calls. This target can. E. coli
# aminopeptidase N, PDB 3B34 at 1.30 A, 891 aa with its His tag, so it pads to 896 -- above the
# route's `_TRIATT_FUSED_HIFI_MIN_S = 128` floor and a real protein with an experimental answer.
# Its ColabFold MSA is already on disk from this row's narrow-q work (71e9b8d1ddaa9905.a3m,
# 2.7 MB), so nothing reaches the network.
#
# 20 sampling steps rather than the CLI default 200, matched across arms, for turn budget: that
# makes the ABSOLUTE distance to 3B34 worse than the 0.60-0.78 A the 200-step narrow-q runs
# recorded, and leaves the off-vs-on DELTA -- which is the measurement -- intact. Stated rather
# than hidden.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/pepnhifi
MSA=$WT/perf/land_standing/out/narrowq_openbind_pepn/msa
IN=$WT/perf/land_standing/fixtures/pepn_3b34.yaml
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
  grep -o '"plddt": [0-9.]*\|"n_tokens": [0-9]*' \
    "$OUT/res_$tag"/openfold3_results_*/results.json 2>/dev/null | tr '\n' ' '
  echo
}

run off  0
run on   1
run off2 0
echo PEPN_DONE

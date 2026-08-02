#!/bin/bash
# How many of the 164 AbAg-XM targets actually fold on a 12 GiB Wormhole chip at FULL uncapped
# unpaired depth, with the code as it stands? Every coverage number in this campaign so far is a
# projection, and all of them have been withdrawn (the footprint model was fitted to capped runs,
# and the module it modelled turned out not to be the one opendde executes). This measures it.
#
# Deliberately runs the DEFAULT gates -- the >1 GiB chunking thresholds -- so the answer describes
# the code a production run would actually use. Targets above the threshold take the chunked path
# and are reported separately, because that path is known to diverge numerically and must not be
# counted as usable.
#
# One fold per (target, chip), 1 diffusion sample: capacity is decided by the trunk, which runs
# before diffusion, so a single sample is enough to answer "does it fit". Skips logical chips
# 0/4/7/15 -- those hung in device bring-up under concurrency and wedged (see the state doc).
#
# Usage: galaxy_capacity_census.sh [n_parallel_chips]
set -u
SRC=${CENSUS_SRC:-$HOME/mthuening/parity-src}
OUT=${CENSUS_OUT:-$HOME/mthuening/census}
MSA=${CENSUS_MSA:-$HOME/mthuening/abag_xm/msa_cache}
CHIPS=${CHIPS:-"1 2 3 5 6 8 9 10 11 12 13 14 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"}

mkdir -p "$OUT"
cd "$SRC" || exit 1
export PYTHONPATH=$SRC
# 28 concurrent folds on 64 cores; omitting a cap has collapsed throughput three times here.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

if [ -n "${CENSUS_TARGETS:-}" ]; then
  read -r -a TARGETS <<< "$CENSUS_TARGETS"
else
  mapfile -t TARGETS < <(ls examples/abag_xm/*.yaml | xargs -n1 basename | sed 's/\.yaml$//' | sort)
fi
# Default: chunking OFF. 125 of 164 targets exceed the 1 GiB gate, so leaving it on would measure
# the numerically-divergent chunked path instead of the capacity of the path a publishable dataset
# can actually use. Set CENSUS_CHUNK_BUDGET to override.
export TT_BIO_MSA_ROW_CHUNK_BUDGET_BYTES=${CENSUS_CHUNK_BUDGET:-999999999999}
echo "targets=${#TARGETS[@]} chips=$(echo $CHIPS | wc -w)"

# Deal the targets round-robin onto chips, then run each chip's list sequentially.
i=0
for c in $CHIPS; do
  list=""
  j=0
  for t in "${TARGETS[@]}"; do
    if [ $((j % $(echo $CHIPS | wc -w))) -eq $i ]; then list="$list $t"; fi
    j=$((j+1))
  done
  (
    for t in $list; do
      d=$OUT/$t
      mkdir -p "$d"
      start=$(date +%s)
      TT_VISIBLE_DEVICES=$c timeout 1800 /usr/bin/python3.10 -u -m tt_bio.main predict \
        "examples/abag_xm/$t.yaml" --model opendde-abag --out_dir "$d" --override \
        --diffusion_samples 1 --max_parallel_samples 1 --seed 42 --host_threads 2 \
        --msa_dir "$MSA" --msa_cache_only > "$OUT/$t.log" 2>&1
      rc=$?
      secs=$(( $(date +%s) - start ))
      cif=$(ls "$d"/opendde_results_$t/structures/*.cif 2>/dev/null | head -1)
      if [ -n "$cif" ]; then verdict=ok; else verdict=fail; fi
      err=$(grep -oE "allocate [0-9]+ B DRAM|Out of Memory|msa_cache_only[^\"]{0,80}" "$OUT/$t.log" 2>/dev/null | head -1)
      printf '{"target":"%s","chip":%s,"verdict":"%s","rc":%s,"seconds":%s,"note":"%s"}\n' \
        "$t" "$c" "$verdict" "$rc" "$secs" "${err//\"/}" >> "$OUT/results.jsonl"
    done
  ) &
  i=$((i+1))
done
wait
echo "CENSUS_DONE"
ok=$(grep -c '"verdict":"ok"' "$OUT/results.jsonl" 2>/dev/null || echo 0)
tot=$(wc -l < "$OUT/results.jsonl" 2>/dev/null || echo 0)
echo "folded: $ok / $tot"

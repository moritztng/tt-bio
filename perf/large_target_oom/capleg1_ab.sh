#!/bin/bash
# Capacity leg-1 A/B (release_gate.py CAPACITY_LEGS[0]): protenix-v2 9j4c_abag,
# 50 samples / mps 5 / 6 steps / seed 0 / single_sequence, TT_BIO_DRAM_PEAK census.
# Question: does MAIN also spike in the eager 4-D transition at [1,1095,1120,256]?
# Branch's denser census tags read 8.73 there; main's coarse tags capped at 5.97.
# mainprobe = c077af3 + dram_peak tags bracketing the eager transition body.
# Legs run concurrently on separate cards (0 branch, 1 mainprobe); DRAM is per-card.
set -u
OUT=$HOME/oomfix_capleg1
mkdir -p "$OUT"
export TT_BIO_LEASE_HOLDER=worker:tt-bio-large-target-oom-rootcause
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

leg() { # <src> <card> <tag>
  local src=$1 card=$2 tag=$3
  setsid bash -c "
    cd $src
    s=\$(date +%s)
    PYTHONPATH=$src TT_VISIBLE_DEVICES=$card TT_BIO_DRAM_PEAK=$OUT/$tag.dram.txt \
      timeout -k 30 3000 /usr/bin/python3 -u -m tt_bio.main predict \
      examples/abag_pilot_expansion/9j4c_abag.yaml --model protenix-v2 \
      --sampling_steps 6 --diffusion_samples 50 --max_parallel_samples 5 \
      --seed 0 --single_sequence --write_pae --out_dir $src \
      > $OUT/$tag.log 2>&1
    rc=\$?
    secs=\$(( \$(date +%s) - s ))
    peak=\$(grep -o 'GiB used' -B0 $OUT/$tag.dram.txt 2>/dev/null | wc -l)
    peak=\$(awk '{for(i=2;i<=NF;i++) if (\$i==\"GiB\" && \$(i-1)+0>m) m=\$(i-1)+0} END {printf \"%.3f\", m+0}' $OUT/$tag.dram.txt 2>/dev/null)
    echo \"{\\\"tag\\\":\\\"$tag\\\",\\\"card\\\":$card,\\\"rc\\\":\$rc,\\\"secs\\\":\$secs,\\\"dram_peak_gib\\\":\${peak:-0}}\" >> $OUT/results.jsonl
  " &
  echo "$! $tag" >> "$OUT/live.list"
  sleep 5
}

leg $HOME/oomfix_src   0 branch
leg $HOME/oomfix_main  1 mainprobe
wait
echo "ALL_DONE $(date -Is)" >> "$OUT/results.jsonl"

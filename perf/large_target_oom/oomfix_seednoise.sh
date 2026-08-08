#!/bin/bash
# S4 fold-level equivalence floor: MAIN code, 9ly6, seeds 1 and 2 (seed 42 already folds in
# run3 as od_9ly6_MAIN). The branch-vs-main delta is meaningful only against the same-code
# seed spread. Runs on chips freed by the finished boltz2 cells.
set -u
H=$HOME/mthuening
B=$H/oomfix_seednoise
MSA=$H/abag_xm/msa_cache
mkdir -p "$B"

fold() { # <seed> <chip> <tag>
  local seed=$1 c=$2 tag=$3
  local ob=$B/$tag
  mkdir -p "$ob"
  setsid env C="$c" TAG="$tag" OB="$ob" SEED="$seed" bash -c "
    cd $H/oomfix_main
    s=\$(date +%s)
    TT_VISIBLE_DEVICES=\$C PYTHONPATH=$H/oomfix_main OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      timeout -k 30 3600 /usr/bin/python3.10 -u -m tt_bio.main predict \
      examples/abag_xm/9ly6.yaml --model opendde-abag --out_dir \$OB --override \
      --diffusion_samples 1 --max_parallel_samples 1 --seed \$SEED --host_threads 2 \
      --msa_dir $MSA --msa_cache_only > $B/\$TAG.log 2>&1
    rc=\$?
    secs=\$(( \$(date +%s) - s ))
    cifs=\$(ls \$OB/*results_*/structures/*.cif 2>/dev/null | wc -l)
    echo \"{\\\"tag\\\":\\\"\$TAG\\\",\\\"seed\\\":\$SEED,\\\"chip\\\":\$C,\\\"rc\\\":\$rc,\\\"secs\\\":\$secs,\\\"cifs\\\":\${cifs:-0}}\" >> $B/results.jsonl
  " &
  echo "$! $tag" >> "$B/live.list"
  sleep 8
}

fold 1 12 main_s1
fold 2 13 main_s2
wait
echo "SEEDNOISE_DONE $(date -Is)" >> "$B/results.jsonl"

#!/bin/bash
# Release-gate capacity legs, measured on the card the budget is supposed to protect.
#
# Both legs are run exactly as scripts/release_gate.py CAPACITY_LEGS defines them
# (same yaml/model/samples/mps/steps/seed/--single_sequence/--write_pae), but on a
# 12 GiB Wormhole chip instead of a 31.9 GiB Blackhole p150a. That matters: the gate
# exists to answer "does the largest supported input fit the smallest card in the
# fleet", and a p150a cannot exercise that question -- it has 2.7x the DRAM, so a leg
# that would OOM a Wormhole still passes there and the budget constant is the only
# thing standing between a footprint regression and the campaign exclusions this
# workstream removed.
#
# Leg 1: protenix-v2 9j4c_abag, 50 samples / mps 5 / 6 steps  (sample-scaled footprint)
# Leg 2: opendde-abag 9ivj,      8 samples / mps 2 / 6 steps  (structural-token ceiling)
set -u
H=$HOME/mthuening
B=$H/capleg_wh
SRC=$H/oomfix_src
mkdir -p "$B"

leg() { # <yaml_rel> <model> <samples> <mps> <chip> <tag>
  local y=$1 m=$2 n=$3 mps=$4 c=$5 tag=$6
  local ob=$B/$tag
  mkdir -p "$ob"
  setsid env Y="$y" M="$m" N="$n" MPS="$mps" C="$c" TAG="$tag" OB="$ob" bash -c "
    cd $SRC
    s=\$(date +%s)
    TT_VISIBLE_DEVICES=\$C TT_BIO_DRAM_PEAK=$B/\$TAG.dram.txt \
      PYTHONPATH=$SRC OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      timeout -k 30 5400 /usr/bin/python3.10 -u -m tt_bio.main predict \
      \$Y --model \$M --sampling_steps 6 --diffusion_samples \$N \
      --max_parallel_samples \$MPS --seed 0 --single_sequence --write_pae \
      --out_dir \$OB > $B/\$TAG.log 2>&1
    rc=\$?
    secs=\$(( \$(date +%s) - s ))
    cifs=\$(ls \$OB/*results_*/structures/*.cif 2>/dev/null | wc -l)
    oom=\$(grep -c 'Out of Memory' $B/\$TAG.log 2>/dev/null)
    peak=\$(awk \"{for(i=2;i<=NF;i++) if (\\\$i==\\\"GiB\\\" && \\\$(i-1)+0>m) m=\\\$(i-1)+0} END {printf \\\"%.3f\\\", m+0}\" $B/\$TAG.dram.txt 2>/dev/null)
    echo \"{\\\"tag\\\":\\\"\$TAG\\\",\\\"model\\\":\\\"\$M\\\",\\\"chip\\\":\$C,\\\"rc\\\":\$rc,\\\"secs\\\":\$secs,\\\"cifs\\\":\${cifs:-0},\\\"oom\\\":\${oom:-0},\\\"dram_peak_gib\\\":\${peak:-0}}\" >> $B/results.jsonl
  " &
  echo "$! $tag" >> "$B/live.list"
  sleep 8
}

leg examples/abag_pilot_expansion/9j4c_abag.yaml protenix-v2   50 5 4 capleg1
leg examples/abag_xm/9ivj.yaml                   opendde-abag   8 2 6 capleg2
wait
echo "ALL_LEGS_DONE $(date -Is)" >> "$B/results.jsonl"

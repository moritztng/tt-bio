#!/bin/bash
# Host-concat verification wave (3f6f84c08): the red od cells on the host-assembly
# fix. 9j4c/9i3p are the acceptance reds; 9ivj re-verifies because its 2.13 GiB z
# now also takes the host-concat path. Chips 4/6/7 (0 runs the concat A/B, 1/2
# run3, 3/5 run5 — do not disturb).
set -u
H=$HOME/mthuening
B=$H/oomfix_run6
MSA=$H/abag_xm/msa_cache
mkdir -p "$B"

fold() { # <target> <chip> <timeout_s> <tag>
  local t=$1 c=$2 to=$3 tag=$4
  local ob=$B/$tag
  mkdir -p "$ob"
  setsid env T="$t" C="$c" TO="$to" TAG="$tag" OB="$ob" bash -c "
    cd $H/oomfix_src
    s=\$(date +%s)
    TT_VISIBLE_DEVICES=\$C TT_BIO_DRAM_PEAK=$B/\$TAG.dram.txt \
      PYTHONPATH=$H/oomfix_src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      timeout -k 30 \$TO /usr/bin/python3.10 -u -m tt_bio.main predict \
      examples/abag_xm/\$T.yaml --model opendde-abag --out_dir \$OB --override \
      --diffusion_samples 1 --max_parallel_samples 1 --seed 42 --host_threads 2 \
      --msa_dir $MSA --msa_cache_only > $B/\$TAG.log 2>&1
    rc=\$?
    secs=\$(( \$(date +%s) - s ))
    cifs=\$(ls \$OB/*results_*/structures/*.cif 2>/dev/null | wc -l)
    oom=\$(grep -c \"Out of Memory\" $B/\$TAG.log 2>/dev/null)
    peak=\$(awk \"{for(i=2;i<=NF;i++) if (\\\$i==\\\"GiB\\\" && \\\$(i-1)+0>m) m=\\\$(i-1)+0} END {printf \\\"%.3f\\\", m+0}\" $B/\$TAG.dram.txt 2>/dev/null)
    echo \"{\\\"tag\\\":\\\"\$TAG\\\",\\\"model\\\":\\\"opendde-abag\\\",\\\"chip\\\":\$C,\\\"rc\\\":\$rc,\\\"secs\\\":\$secs,\\\"cifs\\\":\${cifs:-0},\\\"oom\\\":\${oom:-0},\\\"dram_peak_gib\\\":\${peak:-0}}\" >> $B/results.jsonl
  " &
  echo "$! $to $tag" >> "$B/live.list"
  sleep 8
}

fold 9j4c 4 6000 od_9j4c
fold 9i3p 6 6000 od_9i3p
fold 9ivj 7 6000 od_9ivj
fold 9q7y 2 6000 od_9q7y
wait
echo "ALL_FOLDS_DONE $(date -Is)" >> "$B/results.jsonl"

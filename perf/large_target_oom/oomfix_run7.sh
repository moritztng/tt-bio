#!/bin/bash
# Pair-cond row-block verification wave: run6 showed the host-concat stack holds the
# structural pairformer flat (3.77-3.87 GiB, maxfree steady) but od_9i3p died one stage
# later at the DiffusionConditioning channel concat ([1902,1920,256] bf16 = 1.87 GiB,
# refused at 7.5 used / 4.5 free). PAIRCOND_BLOCK_BYTES row-blocks that chain past
# 2.5 GiB of z. 9i3p/9j4c (2.59/3.19 GiB z) take the blocked path; 9ivj/9q7y (2.10/2.01)
# stay on the run6-verified path and are NOT rerun.
set -u
H=$HOME/mthuening
B=$H/oomfix_run7
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

fold 9i3p 6 6000 od_9i3p
fold 9j4c 2 6000 od_9j4c
wait
echo "ALL_FOLDS_DONE $(date -Is)" >> "$B/results.jsonl"

#!/bin/bash
# run8: PROVE the 9j4c mechanism — is the fp32 diffusion resident the extra ~2.8 GiB that
# pushes the residue-scale CONFIDENCE pairformer over the 12 GiB wall? Two legs, same chip
# sequence as run7, same seed 42, only PROTENIX_DIFFUSION_FP32_DEVICE differs:
#   A: fp32 diffusion (the run7 failing config)   B: bf16 diffusion (PROTENIX_DIFFUSION_FP32_DEVICE=0)
# If B folds 9j4c and its confidence pairformer peak drops ~2.8 GiB below A's 10.04/10.92,
# the mechanism is confirmed and the fix lever is chosen (halve the resident, not chunk the pairformer).
set -u
H=$HOME/mthuening
B=$H/oomfix_run8
MSA=$H/abag_xm/msa_cache
mkdir -p "$B"

fold() { # <target> <chip> <timeout_s> <tag> <fp32>
  local t=$1 c=$2 to=$3 tag=$4 fp32=$5
  local ob=$B/$tag
  mkdir -p "$ob"
  setsid env T="$t" C="$c" TO="$to" TAG="$tag" OB="$ob" FP32="$fp32" bash -c "
    cd $H/oomfix_src
    s=\$(date +%s)
    TT_VISIBLE_DEVICES=\$C TT_BIO_DRAM_PEAK=$B/\$TAG.dram.txt \
      PROTENIX_DIFFUSION_FP32_DEVICE=\$FP32 \
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
    echo \"{\\\"tag\\\":\\\"\$TAG\\\",\\\"model\\\":\\\"opendde-abag\\\",\\\"chip\\\":\$C,\\\"fp32\\\":\$FP32,\\\"rc\\\":\$rc,\\\"secs\\\":\$secs,\\\"cifs\\\":\${cifs:-0},\\\"oom\\\":\${oom:-0},\\\"dram_peak_gib\\\":\${peak:-0}}\" >> $B/results.jsonl
  " &
  echo "$! $to $tag" >> "$B/live.list"
  sleep 8
}

fold 9j4c 2 6000 j4c_fp32 1
fold 9j4c 6 6000 j4c_bf16 0
wait
echo "ALL_FOLDS_DONE $(date -Is)" >> "$B/results.jsonl"

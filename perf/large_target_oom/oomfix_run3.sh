#!/bin/bash
# Final-code acceptance wave for tt-bio-large-target-oom-rootcause: every matrix cell on
# branch HEAD (2eefddd40), so no cell carries a code-state asterisk. od_9j4c/od_9i3p are
# already in flight on HEAD in oomfix_run2 (chips 17/18) and are not duplicated here.
# 17 folds, one per chip, all parallel. Chips skip 0/4/7/15 (wedge-prone) and 17/18 (busy).
set -u
H=$HOME/mthuening
BRANCH_SRC=$H/oomfix_src
MAIN_SRC=$H/oomfix_main
ESM_DEPS=$H/oomfix_deps
B=$H/oomfix_run3
MSA=$H/abag_xm/msa_cache
mkdir -p "$B"
: > "$B/live.list"

fold() { # <src> <model> <target> <chip> <timeout_s> <tag> [esm]
  local src=$1 m=$2 t=$3 c=$4 to=$5 tag=$6 esm=${7:-}
  local ob=$B/$tag
  rm -rf "$ob" "$B/$tag.log" "$B/$tag.dram.txt"
  mkdir -p "$ob"
  if [ -n "$esm" ]; then
    # Campaign config for esmfold2 (p28_fleet.sh): single-sequence, no msa flags,
    # recycling 10 / sampling 100, seed 50000; oomfix_deps pins torchvision.
    setsid env SRC="$src" M="$m" T="$t" C="$c" TO="$to" TAG="$tag" OB="$ob" B="$B" bash -c '
      cd "$SRC"
      s=$(date +%s)
      TT_VISIBLE_DEVICES=$C TT_BIO_DRAM_PEAK=$B/$TAG.dram.txt \
        PYTHONPATH=$HOME/mthuening/oomfix_deps:$SRC OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        timeout -k 30 $TO /usr/bin/python3.10 -u -m tt_bio.main predict \
        examples/abag_xm/$T.yaml --model $M --out_dir $OB --override \
        --diffusion_samples 1 --recycling_steps 10 --sampling_steps 100 --seed 50000 \
        --host_threads 2 > $B/$TAG.log 2>&1
      rc=$?
      secs=$(( $(date +%s) - s ))
      if [ $rc -eq 124 ]; then pkill -TERM -g $$ -f python3.10 2>/dev/null; sleep 5; pkill -KILL -g $$ -f python3.10 2>/dev/null; fi
      cifs=$(ls $OB/*results_*/structures/*.cif 2>/dev/null | wc -l)
      oom=$(grep -c "Out of Memory" $B/$TAG.log 2>/dev/null)
      peak=$(awk "{for(i=2;i<=NF;i++) if (\$i==\"GiB\" && \$(i-1)+0>m) m=\$(i-1)+0} END {printf \"%.3f\", m+0}" $B/$TAG.dram.txt 2>/dev/null)
      echo "{\"tag\":\"$TAG\",\"model\":\"$M\",\"target\":\"$T\",\"chip\":$C,\"rc\":$rc,\"secs\":$secs,\"cifs\":${cifs:-0},\"oom\":${oom:-0},\"dram_peak_gib\":${peak:-0}}" >> $B/results.jsonl
    ' &
  else
    setsid env SRC="$src" M="$m" T="$t" C="$c" TO="$to" TAG="$tag" OB="$ob" B="$B" MSA="$MSA" bash -c '
      cd "$SRC"
      s=$(date +%s)
      TT_VISIBLE_DEVICES=$C TT_BIO_DRAM_PEAK=$B/$TAG.dram.txt \
        PYTHONPATH=$SRC OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        timeout -k 30 $TO /usr/bin/python3.10 -u -m tt_bio.main predict \
        examples/abag_xm/$T.yaml --model $M --out_dir $OB --override \
        --diffusion_samples 1 --max_parallel_samples 1 --seed 42 --host_threads 2 \
        --msa_dir $MSA --msa_cache_only > $B/$TAG.log 2>&1
      rc=$?
      secs=$(( $(date +%s) - s ))
      if [ $rc -eq 124 ]; then pkill -TERM -g $$ -f python3.10 2>/dev/null; sleep 5; pkill -KILL -g $$ -f python3.10 2>/dev/null; fi
      cifs=$(ls $OB/*results_*/structures/*.cif 2>/dev/null | wc -l)
      oom=$(grep -c "Out of Memory" $B/$TAG.log 2>/dev/null)
      peak=$(awk "{for(i=2;i<=NF;i++) if (\$i==\"GiB\" && \$(i-1)+0>m) m=\$(i-1)+0} END {printf \"%.3f\", m+0}" $B/$TAG.dram.txt 2>/dev/null)
      echo "{\"tag\":\"$TAG\",\"model\":\"$M\",\"target\":\"$T\",\"chip\":$C,\"rc\":$rc,\"secs\":$secs,\"cifs\":${cifs:-0},\"oom\":${oom:-0},\"dram_peak_gib\":${peak:-0}}" >> $B/results.jsonl
    ' &
  fi
  echo "$! $to $tag" >> "$B/live.list"
  sleep 8
}

fold $BRANCH_SRC opendde-abag 9ivj 1  6000 od_9ivj
fold $BRANCH_SRC opendde-abag 9q7y 2  6000 od_9q7y
fold $BRANCH_SRC opendde-abag 9mze 3  3600 od_9mze
fold $BRANCH_SRC opendde-abag 9ly6 5  3600 od_9ly6
fold $MAIN_SRC   opendde-abag 9ly6 6  3600 od_9ly6_MAIN
fold $BRANCH_SRC protenix-v2  9j4c 8  3600 px_9j4c
fold $BRANCH_SRC protenix-v2  9i3p 9  3600 px_9i3p
fold $BRANCH_SRC protenix-v2  9ivj 10 3600 px_9ivj
fold $BRANCH_SRC protenix-v2  9q7y 11 3600 px_9q7y
fold $BRANCH_SRC boltz2       9j4c 12 2400 bz_9j4c
fold $BRANCH_SRC boltz2       9i3p 13 2400 bz_9i3p
fold $BRANCH_SRC boltz2       9ivj 14 2400 bz_9ivj
fold $BRANCH_SRC boltz2       9q7y 16 2400 bz_9q7y
fold $BRANCH_SRC esmfold2     9j4c 19 2400 esm_9j4c esm
fold $BRANCH_SRC esmfold2     9i3p 20 2400 esm_9i3p esm
fold $BRANCH_SRC esmfold2     9ivj 21 2400 esm_9ivj esm
fold $BRANCH_SRC esmfold2     9q7y 22 2400 esm_9q7y esm

wait
echo "ALL_FOLDS_DONE $(date -Is)" >> "$B/results.jsonl"

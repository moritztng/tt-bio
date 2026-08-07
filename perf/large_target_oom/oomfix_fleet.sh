#!/bin/bash
# WH Galaxy acceptance window for tt-bio-large-target-oom-rootcause.
# 19 folds, one per chip, all parallel: the 4x4 capability matrix on the fixed branch,
# opendde controls (9mze, 9ly6), and the 9ly6 main-code leg for fold-level equivalence.
# Discipline per galaxy-campaign memory: setsid per fold + pgid kill on timeout, never a
# bare `timeout` (it orphans mp grandchildren that hold the chip).
set -u
H=$HOME/mthuening
BRANCH_SRC=$H/oomfix_src
MAIN_SRC=$H/oomfix_main
B=$H/oomfix_run
MSA=$H/abag_xm/msa_cache
mkdir -p "$B"
: > "$B/live.list"

fold() { # <src> <model> <target> <chip> <timeout_s> <tag>
  local src=$1 m=$2 t=$3 c=$4 to=$5 tag=$6
  local ob=$B/$tag
  rm -rf "$ob" "$B/$tag.log" "$B/$tag.dram.txt"   # no stale cifs/logs from an earlier run
  mkdir -p "$ob"
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
  echo "$! $to $tag" >> "$B/live.list"
}

# chip assignments skip 0/4/7/15 (wedge-prone in bring-up under concurrency, census.sh note)
fold $BRANCH_SRC opendde-abag 9j4c 1  6000 od_9j4c;    sleep 8
fold $BRANCH_SRC opendde-abag 9i3p 2  6000 od_9i3p;    sleep 8
fold $BRANCH_SRC opendde-abag 9ivj 3  6000 od_9ivj;    sleep 8
fold $BRANCH_SRC opendde-abag 9q7y 5  6000 od_9q7y;    sleep 8
fold $BRANCH_SRC opendde-abag 9ly6 8  3600 od_9ly6;    sleep 8
fold $MAIN_SRC   opendde-abag 9ly6 6  3600 od_9ly6_MAIN; sleep 8
fold $BRANCH_SRC opendde-abag 9mze 9  3600 od_9mze;    sleep 8
fold $BRANCH_SRC protenix-v2  9j4c 10 3600 px_9j4c;    sleep 8
fold $BRANCH_SRC protenix-v2  9i3p 11 3600 px_9i3p;    sleep 8
fold $BRANCH_SRC protenix-v2  9ivj 12 3600 px_9ivj;    sleep 8
fold $BRANCH_SRC protenix-v2  9q7y 13 3600 px_9q7y;    sleep 8
# esmfold2 runs from oomfix_esm_fleet.sh instead: it needs the oomfix_deps PYTHONPATH
# (torchvision pin) and the campaign config (no msa flags, recycling 10 / sampling 100).
fold $BRANCH_SRC boltz2       9j4c 19 2400 bz_9j4c;    sleep 8
fold $BRANCH_SRC boltz2       9i3p 20 2400 bz_9i3p;    sleep 8
fold $BRANCH_SRC boltz2       9ivj 21 2400 bz_9ivj;    sleep 8
fold $BRANCH_SRC boltz2       9q7y 22 2400 bz_9q7y

wait
echo "ALL_FOLDS_DONE $(date -Is)" >> "$B/results.jsonl"

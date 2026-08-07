#!/bin/bash
# esmfold2 relaunch: system python hits the broken torchvision 0.23 in ~/.local
# (transformers image_utils imports it during the biohub/ESMFold2 hub load).
# ~/mthuening/oomfix_deps carries torchvision 0.27.0+cpu, the pair for torch 2.12.0+cpu,
# installed with --target (shared envs untouched), per full_parity_gate.py's gatedeps pattern.
set -u
H=$HOME/mthuening
BRANCH_SRC=$H/oomfix_src
B=$H/oomfix_run
MSA=$H/abag_xm/msa_cache

fold() { # <target> <chip>
  local t=$1 c=$2 tag=esm_$1
  rm -rf "$B/$tag" "$B/$tag.log" "$B/$tag.dram.txt"
  mkdir -p "$B/$tag"
  # Campaign config for esmfold2 (p28_fleet.sh fold_esm): single-sequence, NEVER msa
  # flags (they activate the optional MSA encoder on the full cached MSA -- a different,
  # much larger regime that is not what the campaign measures), recycling 10 / sampling
  # 100 explicit, seed 50000. The first oomfix fleet run passed --msa_dir and OOM'd all
  # four targets on MSA-encoder allocations; that was a config artifact, not capability.
  setsid env SRC="$BRANCH_SRC" T="$t" C="$c" TAG="$tag" B="$B" bash -c '
    cd "$SRC"
    s=$(date +%s)
    TT_VISIBLE_DEVICES=$C TT_BIO_DRAM_PEAK=$B/$TAG.dram.txt \
      PYTHONPATH=$HOME/mthuening/oomfix_deps:$SRC OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      timeout -k 30 2400 /usr/bin/python3.10 -u -m tt_bio.main predict \
      examples/abag_xm/$T.yaml --model esmfold2 --out_dir $B/$TAG --override \
      --diffusion_samples 1 --recycling_steps 10 --sampling_steps 100 --seed 50000 \
      --host_threads 2 > $B/$TAG.log 2>&1
    rc=$?
    secs=$(( $(date +%s) - s ))
    if [ $rc -eq 124 ]; then pkill -TERM -g $$ -f python3.10 2>/dev/null; sleep 5; pkill -KILL -g $$ -f python3.10 2>/dev/null; fi
    cifs=$(ls $B/$TAG/*results_*/structures/*.cif 2>/dev/null | wc -l)
    oom=$(grep -c "Out of Memory" $B/$TAG.log 2>/dev/null)
    peak=$(awk "{for(i=2;i<=NF;i++) if (\$i==\"GiB\" && \$(i-1)+0>m) m=\$(i-1)+0} END {printf \"%.3f\", m+0}" $B/$TAG.dram.txt 2>/dev/null)
    echo "{\"tag\":\"$TAG\",\"model\":\"esmfold2\",\"target\":\"$T\",\"chip\":$C,\"rc\":$rc,\"secs\":$secs,\"cifs\":${cifs:-0},\"oom\":${oom:-0},\"dram_peak_gib\":${peak:-0}}" >> $B/results.jsonl
  ' &
  sleep 8
}

fold 9j4c 14
fold 9i3p 16
fold 9ivj 17
fold 9q7y 18
wait
echo "ESM_DONE $(date -Is)" >> "$B/results.jsonl"

#!/bin/bash
# AbAg-XM deep-N saturation, GALAXY window 2b (workstream abag-xm-deepn-saturation-fullpanel).
#
# FULL-PANEL N=64 for protenix-v2 + esmfold2, 148 targets (164 minus the 16-target pilot).
# GATED: launch ONLY after scripts/abag_xm/phase0_n64_gate.py records LICENSED for BOTH
# models (pre-registered N=64 cross-hardware gate, state doc PHASE 2). If either model
# verdicts STOP, delete its tasks before launching.
#   protenix-v2  148 targets x N=64, seed 30000, mps 5 narrowing 5->2->1 on OOM
#   esmfold2     148 targets x N=64, seed 50000, exact p2 config (single-seq, venv python,
#                recycling 10 / sampling 100 explicit, engine auto chunking)
#
# Every fold appends one JSON record to results.jsonl (per-attempt records for narrowing).
# DONE_CHECK convention: no literal percent strings in logs.
set -u
H=$HOME/mthuening
SRC=$H/deepn_src
B=$H/p26b; mkdir -p $B $B/claims
NCHIP=${1:-32}
STAGGER=${2:-8}
PY_SYS=/usr/bin/python3.10
PY_VENV=$H/tt-bio/env/bin/python3.10

PILOT16="21tw 9d3j 9i3p 9j4c 9ly5 9m0j 9ma0 9obn 9ppw 9q6y 9rye 9ua5 9udq 9v0x 9wpm 9zen"

TASKS=$B/tasks.txt
{
  for y in $SRC/examples/abag_xm/*.yaml; do
    t=$(basename $y .yaml)
    skip=0
    for p in $PILOT16; do [ "$t" = "$p" ] && skip=1; done
    [ $skip = 1 ] && continue
    echo "protenix-v2 $t 64 30000"
    echo "esmfold2 $t 64 50000"
  done
} > $TASKS
echo "tasks: $(wc -l < $TASKS)  chips: $NCHIP"

record() {  # record <model> <target> <rung> <seed> <mps> <chip> <rc> <secs> <cifs> <distinct> <oom>
  printf '{"model":"%s","target":"%s","rung":%s,"seed":%s,"mps":"%s","umd":%s,"rc":%s,"seconds":%s,"cifs":%s,"distinct":%s,"oom":%s}\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" >> $B/results.jsonl
}

count_structs() { # <dir> -> echoes "n distinct"
  local d=$1 n distinct
  n=$(ls $d/structures/*.cif 2>/dev/null | wc -l)
  distinct=$(md5sum $d/structures/*.cif 2>/dev/null | awk '{print $1}' | sort -u | wc -l)
  echo "${n:-0} ${distinct:-0}"
}

fold_px() { # <target> <rung> <seed> <chip>  -- mps narrowing 5->2->1 on OOM
  local t=$1 rung=$2 seed=$3 u=$4 mps s rc secs oom d nd
  for mps in 5 2 1; do
    s=$(date +%s)
    timeout 7200 env TT_VISIBLE_DEVICES=$u $PY_SYS -u -m tt_bio.main predict \
      examples/abag_xm/$t.yaml --model protenix-v2 --out_dir $B/protenix/$t --override \
      --diffusion_samples $rung --max_parallel_samples $mps --seed $seed --host_threads 2 \
      --msa_dir $MSA --msa_cache_only > $B/protenix_${t}_mps$mps.log 2>&1
    rc=$?; secs=$(( $(date +%s) - s ))
    d=$(ls -d $B/protenix/$t/*results_$t 2>/dev/null | head -1)
    read -r _n _di <<<$(count_structs "$d")
    oom=$(grep -c 'Out of Memory' $B/protenix_${t}_mps$mps.log 2>/dev/null)
    record protenix-v2 $t $rung $seed $mps $u $rc $secs ${_n:-0} ${_di:-0} ${oom:-0}
    [ "${oom:-0}" -gt 0 ] && [ "${_n:-0}" -eq 0 ] && [ "$mps" != 1 ] || break
  done
}

fold_esm() { # <target> <rung> <seed> <chip>  -- exact p2 config, single-sequence
  local t=$1 rung=$2 seed=$3 u=$4 s rc secs oom d nd
  s=$(date +%s)
  timeout 7200 env TT_VISIBLE_DEVICES=$u $PY_VENV -u -m tt_bio.main predict \
    examples/abag_xm/$t.yaml --model esmfold2 --out_dir $B/esmfold2/$t --override \
    --diffusion_samples $rung --recycling_steps 10 --sampling_steps 100 --seed $seed \
    --host_threads 2 > $B/esmfold2_$t.log 2>&1
  rc=$?; secs=$(( $(date +%s) - s ))
  d=$(ls -d $B/esmfold2/$t/*results_$t 2>/dev/null | head -1)
  read -r _n _di <<<$(count_structs "$d")
  oom=$(grep -c 'Out of Memory' $B/esmfold2_$t.log 2>/dev/null)
  record esmfold2 $t $rung $seed auto $u $rc $secs ${_n:-0} ${_di:-0} ${oom:-0}
}

fold() { # <model> <target> <rung> <seed> <chip>
  case "$1" in
    protenix-v2) fold_px "$2" "$3" "$4" "$5";;
    esmfold2) fold_esm "$2" "$3" "$4" "$5";;
  esac
}

slot() {
  local chip=$1 idx n model t rung seed
  n=$(wc -l < $TASKS)
  for ((idx=1; idx<=n; idx++)); do
    mkdir $B/claims/$idx 2>/dev/null || continue
    read -r model t rung seed <<<"$(sed -n "${idx}p" $TASKS)"
    ( cd $SRC && fold "$model" "$t" "$rung" "$seed" "$chip" )
  done
  echo "slot $chip done" >> $B/slots.log
}

for (( c=0; c<NCHIP; c++ )); do
  slot "$c" &
  sleep "$STAGGER"
done
wait
echo P26B_DONE >> $B/results.jsonl

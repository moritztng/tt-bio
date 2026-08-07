#!/bin/bash
# AbAg-XM deep-N saturation, GALAXY window 2 (workstream abag-xm-deepn-saturation-fullpanel).
#
# FULL-PANEL N=64 for the PHASE-0-licensed models, 148 targets (164 minus the 16-target
# pilot, which already holds N=64 at identical seeds/config/hardware):
#   boltz2       148 targets x N=64, seed 40000, mps 1 (p2 WH-proven config)
#   opendde-abag 148 targets x N=64, seed 20000, mps 5 narrowing 5->2->1 on OOM
# protenix-v2/esmfold2 ride in p26b AFTER the pre-registered N=64 cross-hardware gate.
#
# Every fold appends one JSON record to results.jsonl (per-attempt records for od narrowing).
# DONE_CHECK convention: no literal percent strings in logs.
set -u
H=$HOME/mthuening
SRC=$H/deepn_src
B=$H/p26; mkdir -p $B $B/claims
NCHIP=${1:-32}
STAGGER=${2:-8}
PY_SYS=/usr/bin/python3.10
MSA=$H/abag_xm/msa_cache

PILOT16="21tw 9d3j 9i3p 9j4c 9ly5 9m0j 9ma0 9obn 9ppw 9q6y 9rye 9ua5 9udq 9v0x 9wpm 9zen"

TASKS=$B/tasks.txt
{
  for y in $SRC/examples/abag_xm/*.yaml; do
    t=$(basename $y .yaml)
    skip=0
    for p in $PILOT16; do [ "$t" = "$p" ] && skip=1; done
    [ $skip = 1 ] && continue
    echo "boltz2 $t 64 40000"
    echo "opendde-abag $t 64 20000"
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

fold_bz() { # <target> <rung> <seed> <chip>
  local t=$1 rung=$2 seed=$3 u=$4 s rc secs oom d nd
  s=$(date +%s)
  timeout 10800 env TT_VISIBLE_DEVICES=$u $PY_SYS -u -m tt_bio.main predict \
    examples/abag_xm/$t.yaml --model boltz2 --out_dir $B/boltz2/$t --override \
    --diffusion_samples $rung --max_parallel_samples 1 --seed $seed --host_threads 2 \
    --msa_dir $MSA --msa_cache_only > $B/boltz2_$t.log 2>&1
  rc=$?; secs=$(( $(date +%s) - s ))
  d=$(ls -d $B/boltz2/$t/*results_$t 2>/dev/null | head -1)
  read -r _n _di <<<$(count_structs "$d")
  oom=$(grep -c 'Out of Memory' $B/boltz2_$t.log 2>/dev/null)
  record boltz2 $t $rung $seed 1 $u $rc $secs ${_n:-0} ${_di:-0} ${oom:-0}
}

fold_od() { # <target> <rung> <seed> <chip>  -- mps narrowing 5->2->1 on OOM
  local t=$1 rung=$2 seed=$3 u=$4 mps s rc secs oom d nd
  for mps in 5 2 1; do
    s=$(date +%s)
    timeout 10800 env TT_VISIBLE_DEVICES=$u $PY_SYS -u -m tt_bio.main predict \
      examples/abag_xm/$t.yaml --model opendde-abag --out_dir $B/opendde/$t --override \
      --diffusion_samples $rung --max_parallel_samples $mps --seed $seed --host_threads 2 \
      --msa_dir $MSA --msa_cache_only > $B/opendde_${t}_mps$mps.log 2>&1
    rc=$?; secs=$(( $(date +%s) - s ))
    d=$(ls -d $B/opendde/$t/*results_$t 2>/dev/null | head -1)
    read -r _n _di <<<$(count_structs "$d")
    oom=$(grep -c 'Out of Memory' $B/opendde_${t}_mps$mps.log 2>/dev/null)
    record opendde-abag $t $rung $seed $mps $u $rc $secs ${_n:-0} ${_di:-0} ${oom:-0}
    [ "${oom:-0}" -gt 0 ] && [ "${_n:-0}" -eq 0 ] && [ "$mps" != 1 ] || break
  done
}

fold() { # <model> <target> <rung> <seed> <chip>
  case "$1" in
    boltz2) fold_bz "$2" "$3" "$4" "$5";;
    opendde-abag) fold_od "$2" "$3" "$4" "$5";;
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
echo P26_DONE >> $B/results.jsonl

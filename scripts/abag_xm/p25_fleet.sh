#!/bin/bash
# AbAg-XM deep-N saturation, GALAXY window 1 (workstream abag-xm-deepn-saturation-fullpanel).
#
# Two arms in one shared claim-queue over 32 chips, run with japanfold.service STOPPED
# (maintenance window; galaxy_device_lock held by worker:abag-xm-deepn-saturation-fullpanel):
#
#   CONTROL (PHASE 0 attribution pre-flight, blocking for the px/esm campaign legs):
#     protenix-v2  7 GT targets x N=16, seed 30000, mps 5 (BH-matching config; mps-1-vs-5 test)
#     esmfold2     7 GT targets x N=16, seed 50000, exact p2 config (single-seq, recycling 10 /
#                  sampling 100 explicit, venv python; code-drift-vs-arch test)
#     Control set = pilot GT 8 minus 9j4c (WH capacity exclusion, p2-documented, both models).
#
#   PILOT (deep rung for the PHASE-0-consistent models):
#     boltz2       16 pilot targets x N=64, seed 40000, mps 1 (p2 WH-proven config)
#     opendde-abag 16 pilot targets x N=64, seed 20000, mps 5 narrowing 5->2->1 on OOM
#
# Every fold appends one JSON record to results.jsonl (per-attempt records for od narrowing).
# DONE_CHECK convention: no literal percent strings in logs.
set -u
H=$HOME/mthuening
SRC=$H/deepn_src
B=$H/p25; mkdir -p $B $B/claims
NCHIP=${1:-32}
STAGGER=${2:-8}
PY_SYS=/usr/bin/python3.10
PY_VENV=$H/tt-bio/env/bin/python3.10
MSA=$H/abag_xm/msa_cache

CTRL="21tw 9d3j 9ma0 9obn 9q6y 9udq 9wpm"
PILOT_REST="9ma0 9obn 9q6y 9udq 9wpm 21tw 9d3j 9ly5 9m0j 9ppw 9rye 9ua5 9v0x 9zen"

TASKS=$B/tasks.txt
{
  echo "boltz2 9j4c 64 40000"      # longest first: ~90 s/sample at N=64 on WH
  echo "opendde-abag 9j4c 64 20000"  # expected WH capacity exclusion (documented p2)
  echo "boltz2 9i3p 64 40000"
  echo "opendde-abag 9i3p 64 20000"  # documented p2 opendde exclusion
  for t in $CTRL; do
    echo "protenix-v2 $t 16 30000"   # control wave: decision-blocking, quick
    echo "esmfold2 $t 16 50000"
  done
  for t in $PILOT_REST; do
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

fold_px() { # <target> <rung> <seed> <chip>  -- control: mps 5, fallback mps 1 only on OOM
  local t=$1 rung=$2 seed=$3 u=$4 mps s rc secs oom d nd
  for mps in 5 1; do
    s=$(date +%s)
    timeout 3600 env TT_VISIBLE_DEVICES=$u $PY_SYS -u -m tt_bio.main predict \
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

fold_esm() { # <target> <rung> <seed> <chip>  -- control: exact p2 config, single-sequence
  local t=$1 rung=$2 seed=$3 u=$4 s rc secs oom d nd
  s=$(date +%s)
  timeout 3600 env TT_VISIBLE_DEVICES=$u $PY_VENV -u -m tt_bio.main predict \
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
    boltz2) fold_bz "$2" "$3" "$4" "$5";;
    opendde-abag) fold_od "$2" "$3" "$4" "$5";;
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
echo P25_DONE >> $B/results.jsonl

#!/bin/bash
# AbAg-XM deep-N saturation, GALAXY window 3 (workstream abag-xm-deepn-saturation-fullpanel).
#
# Arms in one shared claim-queue over 32 chips, maintenance window (japanfold.service stopped,
# galaxy_device_lock held by worker:abag-xm-deepn-saturation-fullpanel):
#
#   RETRY SWEEP (N=64, completes the panel; single folds, timeout 21600):
#     boltz2 21 + opendde 22 targets that failed p26/p25 as rc=124 (3h cap too small) or
#       rc=1 (transient sqlite store failure on the 99%-full disk) -- NOT capacity.
#       od 9ivj and 9q7y were skipped here as "WH DRAM exclusions". They are not: both fold
#       on this Galaxy (9ivj 1744 s / 9.62 GiB, 9q7y 1716 s / 8.98 GiB, ws:tt-bio-large-
#       target-oom-rootcause). The skip is removed; kept in this comment so the record shows
#       what p27 actually ran.
#     protenix 9 + esmfold2 6 pilot targets that hit p25b's 7200s cap -- these complete the
#       px/esm N=64 overlay the pre-registered cross-hardware gate consumes (gate runs on qb1).
#
#   N=256 RUNG (the deep ladder's next rung for the PHASE-0-licensed models):
#     boltz2 164 targets x 4 chunks x 64 samples, seeds 40000+1000*j, mps 1
#     opendde 160 targets x 4 chunks x 64 samples, seeds 20000+1000*j, mps 5->2->1 narrowing
#       (p27 as run skipped 9i3p 9j4c 9ivj 9q7y as "WH DRAM exclusions at mps=1"; all four
#        fold, so the skip list is empty now and the four cells are owed to the panel)
#     Chunking is RAM-forced: boltz2 holds ~0.22 GB/sample in host RAM (sat-depth: 221 GB per
#     1000-sample fold); 64-sample chunks cap a fold at ~15 GB so 32 concurrent folds fit in
#     the galaxy's 566 GB. Chunk records carry chunk/chunks; harvest pools <t>_c<j> dirs.
#
# Every attempt appends one JSON record to results.jsonl. DONE_CHECK convention: no literal
# percent strings in logs.
set -u
H=$HOME/mthuening
SRC=$H/deepn_src
B=$H/p27; mkdir -p $B $B/claims
NCHIP=${1:-32}
STAGGER=${2:-8}
PY_SYS=/usr/bin/python3.10
PY_VENV=$H/tt-bio/env/bin/python3.10
MSA=$H/abag_xm/msa_cache

BZ_RETRY="21av 22ps 9d72 9d74 9gvn 9hv9 9i5n 9iar 9mze 9mzf 9n09 9nl1 9u5r 9xth 9y0a 9y0e 9yc5 9yio 9ynx 9yxd 9zdu"
OD_RETRY="21av 21du 9d72 9d73 9gfr 9lxp 9ly3 9q6h 9q6n 9u5p 9ve0 9xqn 9xsx 9xth 9y0a 9y0e 9yc5 9yio 9ynx 9yxd 9zdu 9rye"
PX_RETRY="9d3j 9i3p 9j4c 9ly5 9m0j 9ma0 9obn 9udq 9zen"
ESM_RETRY="21tw 9d3j 9j4c 9m0j 9ppw 9zen"
OD_EXCL=""

TASKS=$B/tasks.txt
{
  for t in $BZ_RETRY;  do echo "boltz2 $t 64 40000 0 1"; done
  for t in $OD_RETRY;  do echo "opendde-abag $t 64 20000 0 1"; done
  for t in $PX_RETRY;  do echo "protenix-v2 $t 64 30000 0 1"; done
  for t in $ESM_RETRY; do echo "esmfold2 $t 64 50000 0 1"; done
  for y in $SRC/examples/abag_xm/*.yaml; do
    t=$(basename $y .yaml)
    for j in 0 1 2 3; do
      echo "boltz2 $t 256 $((40000+1000*j)) $j 4"
    done
    skip=0
    for e in $OD_EXCL; do [ "$t" = "$e" ] && skip=1; done
    [ $skip = 1 ] && continue
    for j in 0 1 2 3; do
      echo "opendde-abag $t 256 $((20000+1000*j)) $j 4"
    done
  done
} > $TASKS
echo "tasks: $(wc -l < $TASKS)  chips: $NCHIP"

record() {  # record <model> <target> <rung> <seed> <chunk> <chunks> <mps> <chip> <rc> <secs> <cifs> <distinct> <oom>
  printf '{"model":"%s","target":"%s","rung":%s,"seed":%s,"chunk":%s,"chunks":%s,"mps":"%s","umd":%s,"rc":%s,"seconds":%s,"cifs":%s,"distinct":%s,"oom":%s}\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" >> $B/results.jsonl
}

count_structs() { # <dir> -> echoes "n distinct"
  local d=$1 n distinct
  n=$(ls $d/structures/*.cif 2>/dev/null | wc -l)
  distinct=$(md5sum $d/structures/*.cif 2>/dev/null | awk '{print $1}' | sort -u | wc -l)
  echo "${n:-0} ${distinct:-0}"
}

outbase() { # <model> <target> <chunk> <chunks> -> per-task out dir under $B
  local m=$1 t=$2 c=$3 k=$4
  if [ "$k" -gt 1 ]; then echo "$B/$m/${t}_c$c"; else echo "$B/$m/$t"; fi
}

fold_bz() { # <target> <rung> <seed> <chunk> <chunks> <chip>
  local t=$1 rung=$2 seed=$3 c=$4 k=$5 u=$6 s rc secs oom d nd ob
  ob=$(outbase boltz2 $t $c $k)
  s=$(date +%s)
  timeout 21600 env TT_VISIBLE_DEVICES=$u $PY_SYS -u -m tt_bio.main predict \
    examples/abag_xm/$t.yaml --model boltz2 --out_dir $ob --override \
    --diffusion_samples $((rung/k)) --max_parallel_samples 1 --seed $seed --host_threads 2 \
    --msa_dir $MSA --msa_cache_only > $B/boltz2_${t}_c$c.log 2>&1
  rc=$?; secs=$(( $(date +%s) - s ))
  d=$(ls -d $ob/*results_$t 2>/dev/null | head -1)
  read -r _n _di <<<$(count_structs "$d")
  oom=$(grep -c 'Out of Memory' $B/boltz2_${t}_c$c.log 2>/dev/null)
  record boltz2 $t $rung $seed $c $k 1 $u $rc $secs ${_n:-0} ${_di:-0} ${oom:-0}
}

fold_od() { # <target> <rung> <seed> <chunk> <chunks> <chip>  -- mps narrowing 5->2->1 on OOM
  local t=$1 rung=$2 seed=$3 c=$4 k=$5 u=$6 mps s rc secs oom d nd ob
  ob=$(outbase opendde $t $c $k)
  for mps in 5 2 1; do
    s=$(date +%s)
    timeout 21600 env TT_VISIBLE_DEVICES=$u $PY_SYS -u -m tt_bio.main predict \
      examples/abag_xm/$t.yaml --model opendde-abag --out_dir $ob --override \
      --diffusion_samples $((rung/k)) --max_parallel_samples $mps --seed $seed --host_threads 2 \
      --msa_dir $MSA --msa_cache_only > $B/opendde_${t}_c${c}_mps$mps.log 2>&1
    rc=$?; secs=$(( $(date +%s) - s ))
    d=$(ls -d $ob/*results_$t 2>/dev/null | head -1)
    read -r _n _di <<<$(count_structs "$d")
    oom=$(grep -c 'Out of Memory' $B/opendde_${t}_c${c}_mps$mps.log 2>/dev/null)
    record opendde-abag $t $rung $seed $c $k $mps $u $rc $secs ${_n:-0} ${_di:-0} ${oom:-0}
    [ "${oom:-0}" -gt 0 ] && [ "${_n:-0}" -eq 0 ] && [ "$mps" != 1 ] || break
  done
}

fold_px() { # <target> <rung> <seed> <chunk> <chunks> <chip>  -- mps 5, narrow 5->1 on OOM
  local t=$1 rung=$2 seed=$3 c=$4 k=$5 u=$6 mps s rc secs oom d nd ob
  ob=$(outbase protenix $t $c $k)
  for mps in 5 1; do
    s=$(date +%s)
    timeout 21600 env TT_VISIBLE_DEVICES=$u $PY_SYS -u -m tt_bio.main predict \
      examples/abag_xm/$t.yaml --model protenix-v2 --out_dir $ob --override \
      --diffusion_samples $((rung/k)) --max_parallel_samples $mps --seed $seed --host_threads 2 \
      --msa_dir $MSA --msa_cache_only > $B/protenix_${t}_c${c}_mps$mps.log 2>&1
    rc=$?; secs=$(( $(date +%s) - s ))
    d=$(ls -d $ob/*results_$t 2>/dev/null | head -1)
    read -r _n _di <<<$(count_structs "$d")
    oom=$(grep -c 'Out of Memory' $B/protenix_${t}_c${c}_mps$mps.log 2>/dev/null)
    record protenix-v2 $t $rung $seed $c $k $mps $u $rc $secs ${_n:-0} ${_di:-0} ${oom:-0}
    [ "${oom:-0}" -gt 0 ] && [ "${_n:-0}" -eq 0 ] && [ "$mps" != 1 ] || break
  done
}

fold_esm() { # <target> <rung> <seed> <chunk> <chunks> <chip>  -- single-seq, auto chunking
  local t=$1 rung=$2 seed=$3 c=$4 k=$5 u=$6 s rc secs oom d nd ob
  ob=$(outbase esmfold2 $t $c $k)
  s=$(date +%s)
  timeout 21600 env TT_VISIBLE_DEVICES=$u $PY_VENV -u -m tt_bio.main predict \
    examples/abag_xm/$t.yaml --model esmfold2 --out_dir $ob --override \
    --diffusion_samples $((rung/k)) --recycling_steps 10 --sampling_steps 100 --seed $seed \
    --host_threads 2 > $B/esmfold2_${t}_c$c.log 2>&1
  rc=$?; secs=$(( $(date +%s) - s ))
  d=$(ls -d $ob/*results_$t 2>/dev/null | head -1)
  read -r _n _di <<<$(count_structs "$d")
  oom=$(grep -c 'Out of Memory' $B/esmfold2_${t}_c$c.log 2>/dev/null)
  record esmfold2 $t $rung $seed $c $k auto $u $rc $secs ${_n:-0} ${_di:-0} ${oom:-0}
}

fold() { # <model> <target> <rung> <seed> <chunk> <chunks> <chip>
  case "$1" in
    boltz2)       fold_bz  "$2" "$3" "$4" "$5" "$6" "$7";;
    opendde-abag) fold_od  "$2" "$3" "$4" "$5" "$6" "$7";;
    protenix-v2)  fold_px  "$2" "$3" "$4" "$5" "$6" "$7";;
    esmfold2)     fold_esm "$2" "$3" "$4" "$5" "$6" "$7";;
  esac
}

slot() {
  local chip=$1 idx n model t rung seed c k
  n=$(wc -l < $TASKS)
  for ((idx=1; idx<=n; idx++)); do
    mkdir $B/claims/$idx 2>/dev/null || continue
    read -r model t rung seed c k <<<"$(sed -n "${idx}p" $TASKS)"
    ( cd $SRC && fold "$model" "$t" "$rung" "$seed" "$c" "$k" "$chip" )
  done
  echo "slot $chip done" >> $B/slots.log
}

# NCHIP takes the first N chips; CHIPS names them, for a run that has to skip a wedged or
# co-tenanted one. The Galaxy carried a whole second copy of this script (p27_fleet6.sh) whose
# only difference was a hardcoded `CHIPS="1 4 12 13 14 16"` here, and that copy still had the
# four-target OD_EXCL in it long after this one was cleared. One script, one exclusion list.
for c in ${CHIPS:-$(seq 0 $((NCHIP-1)))}; do
  slot "$c" &
  sleep "$STAGGER"
done
wait
echo P27_DONE >> $B/results.jsonl

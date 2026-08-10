#!/bin/bash
# AbAg-XM deep-N, GALAXY window 7b (workstream abag-xm-deepn-n512): the four formerly
# DRAM-excluded large targets at N=512, on the OOM-fixed engine.
#
# p31 folds the 160/163-target panel on the frozen p27-era engine tree (deepn_src, mtime
# gate, link phase). p32 exists because the large-target cells need engine code that
# contains the large-target OOM fix, and deploying that fix into deepn_src would trip
# p31's link gate and re-fold ~1000 card-h. So p32 is a SEPARATE tree (deepn_src_oomfix,
# branch wk/tt-bio-large-target-oom-rootcause tip, self-consistent) and a separate output
# root. Every chunk here folds fresh: there is nothing to reuse, hence no link phase and
# no mtime gate.
#
# Cells (48 chunk-folds, 64 samples each, seeds base+1000*j, j=0..7 -- the same ladder
# the rest of the panel uses, so these cells are seed-nested across rungs too):
#   opendde-abag  9i3p 9j4c 9ivj 9q7y  x chunks 0-7 = 32  (od never folded these on WH)
#   protenix-v2   9j4c                 x chunks 0-7 =  8
#   esmfold2      9j4c                 x chunks 0-7 =  8  (single-sequence, no MSA flags,
#                                                        no mps -- campaign protocol)
# boltz2 already folded all four targets through p27/p28; its chunk 4-7 are p31 tasks.
#
# Each cell pools 512 samples all folded on the fix-branch engine, so a cell is internally
# single-engine; the engine commit is recorded in $B/engine_commit.txt (the p31 record
# schema is unchanged -- downstream harvest attributes p32 chunks by window).
#
# Chained after p31 by p31_watchdog.sh, which launches this script on P31_DONE and
# respawns the prod worker only after P32_DONE (or this window's own deadline). The
# device lock stays campaign-held across both.
#
# od keeps the campaign-standard mps 5->2->1 narrowing (measured single-sample DRAM peaks
# on the fixed engine: 8.8-9.6 GiB, so mps=5 cannot fit; the narrowing loop finds the fit
# empirically and records it). Runner hardening as p31 (setsid process groups, no-progress
# kill, zero-log kill, post-hang tt-smi -r quarantine, rc=124 kill class) but with
# large-target thresholds: a healthy fold's log stays 0 bytes until completion (p31 pass-5
# lesson), and these chunks are the longest of the campaign -- the handback's smoke-fold
# times (1716-2902 s) scale to ~3.6-6 h per 64-sample od chunk at the p31-measured
# typical-target ratio (2097 s chunk / 280 s smoke = 7.5x). So ZERO_MIN=600 (10 h) sits
# above every plausible legit silent chunk, and CAP_S=43200 (12 h) gives ~2x margin over
# the longest projection; the 45-min no-progress leg (CPU-based) is the real hang
# detector. p31's ZERO_MIN=99 was calibrated on p29's 89-min max silent fold, which does
# NOT cover these cells.
set -u
H=$HOME/mthuening
SRC=$H/deepn_src_oomfix
B=$H/p32; mkdir -p $B $B/claims $B/tries
NCHIP=${1:-32}
STAGGER=${2:-8}
# default excludes chips quarantined during p31 (hang-prone; 18 died to PCI rescan)
CHIPS=${CHIPS:-"1 3 4 5 6 7 8 9 11 12 13 14 15 17 19 20 21 22 23 24 25 26 27 28 29 30 31"}
PY_SYS=/usr/bin/python3.10
PY_VENV=$H/tt-bio/env/bin/python3.10
MSA=$H/abag_xm/msa_cache

OD_T="9i3p 9j4c 9ivj 9q7y"
PX_T="9j4c"
ESM_T="9j4c"

TASKS=$B/tasks.txt
{
  for j in 0 1 2 3 4 5 6 7; do
    for t in $OD_T;  do echo "opendde-abag $t 512 $((20000 + 1000 * j)) $j 8"; done
    for t in $PX_T;  do echo "protenix-v2  $t 512 $((30000 + 1000 * j)) $j 8"; done
    for t in $ESM_T; do echo "esmfold2     $t 512 $((50000 + 1000 * j)) $j 8"; done
  done
} > $TASKS
echo "tasks: $(wc -l < $TASKS)  chips: $(wc -w <<<"$CHIPS") [$CHIPS]"

group_cpu() { # <pgid> -> total CPU seconds of every process in the group
  ps -eo pgid=,times= | awk -v g="$1" '$1==g {s+=$2} END {print s+0}'
}

guarded_fold() { # <logfile> <chip> <cmd...> -- setsid launch + stall/cap group kills + quarantine
  local log=$1 u=$2; shift 2
  local stall_min=${STALL_MIN:-45} cap_s=${CAP_S:-43200} grace_s=${GRACE_S:-120} min_cpu=${MIN_CPU_S:-60}
  local zero_min=${ZERO_MIN:-600}
  local poll_s=${POLL_S:-60}
  setsid env TT_VISIBLE_DEVICES=$u "$@" > "$log" 2>&1 &
  local pid=$! t0=$(date +%s) last_cpu=-1 last_size=-1 stall=0 zero=0 killrc=0 g=0
  while kill -0 $pid 2>/dev/null; do
    sleep $poll_s
    kill -0 $pid 2>/dev/null || break
    local cpu=$(group_cpu $pid) size=$(stat -c %s "$log" 2>/dev/null || echo 0)
    if [ "$size" = "0" ]; then zero=$((zero+1)); else zero=0; fi
    if [ "$last_cpu" -ge 0 ]; then
      if [ $((cpu - last_cpu)) -ge $min_cpu ] || [ "$size" != "$last_size" ]; then
        stall=0
      else
        stall=$((stall+1))
      fi
    fi
    last_cpu=$cpu; last_size=$size
    if [ $zero -ge $zero_min ]; then
      echo "$(date -u +%FT%TZ) GUARD: zero-log kill (${zero}m at 0 bytes, pre-banner spin class) -> INT pgid $pid" >> "$log"
      kill -INT -- -$pid 2>/dev/null
      g=0; while kill -0 $pid 2>/dev/null && [ $g -lt $grace_s ]; do sleep 2; g=$((g+2)); done
      if kill -0 $pid 2>/dev/null; then
        echo "$(date -u +%FT%TZ) GUARD: INT grace expired -> KILL pgid $pid" >> "$log"
        kill -KILL -- -$pid 2>/dev/null
      fi
      killrc=124; break
    fi
    if [ $stall -ge $stall_min ]; then
      echo "$(date -u +%FT%TZ) GUARD: no-progress kill (${stall}m without cpu/log growth) -> INT pgid $pid" >> "$log"
      kill -INT -- -$pid 2>/dev/null
      g=0; while kill -0 $pid 2>/dev/null && [ $g -lt $grace_s ]; do sleep 2; g=$((g+2)); done
      if kill -0 $pid 2>/dev/null; then
        echo "$(date -u +%FT%TZ) GUARD: INT grace expired -> KILL pgid $pid" >> "$log"
        kill -KILL -- -$pid 2>/dev/null
      fi
      killrc=124; break
    fi
    if [ $(( $(date +%s) - t0 )) -ge $cap_s ]; then
      echo "$(date -u +%FT%TZ) GUARD: ${cap_s}s cap -> TERM pgid $pid" >> "$log"
      kill -TERM -- -$pid 2>/dev/null
      g=0; while kill -0 $pid 2>/dev/null && [ $g -lt 60 ]; do sleep 2; g=$((g+2)); done
      kill -0 $pid 2>/dev/null && kill -KILL -- -$pid 2>/dev/null
      killrc=124; break
    fi
  done
  wait $pid 2>/dev/null; local rc=$?
  if [ $killrc -ne 0 ]; then
    rc=$killrc
    sudo -n tt-smi -r /dev/tenstorrent/$u >> "$log" 2>&1 \
      || echo "$(date -u +%FT%TZ) GUARD: tt-smi reset failed on dev $u" >> "$log"
    sleep 10
  fi
  return $rc
}

record() {  # record <model> <target> <rung> <seed> <chunk> <chunks> <mps> <chip> <rc> <secs> <cifs> <distinct> <oom>
  printf '{"model":"%s","target":"%s","rung":%s,"seed":%s,"chunk":%s,"chunks":%s,"mps":"%s","umd":%s,"rc":%s,"seconds":%s,"cifs":%s,"distinct":%s,"oom":%s}\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" >> $B/results.jsonl
  # Mark the claim satisfied, so slot() can tell a completed cell from a killed one. Success is
  # the criterion the link phase and the analysis already use: rc 0 plus a full set of
  # md5-distinct CIFs (rung/chunks of them). CLAIM is exported by slot() into the fold subshell.
  if [ -n "${CLAIM:-}" ] && [ "$9" = 0 ] && [ "${11}" -eq $(( $3 / $6 )) ] && [ "${12}" -eq "${11}" ]; then
    touch "$CLAIM/ok"
  fi
}

count_structs() { # <dir> -> echoes "n distinct"
  local d=$1 n distinct
  n=$(ls $d/structures/*.cif 2>/dev/null | wc -l)
  distinct=$(md5sum $d/structures/*.cif 2>/dev/null | awk '{print $1}' | sort -u | wc -l)
  echo "${n:-0} ${distinct:-0}"
}

outbase() { # <mdir> <target> <chunk> -> per-task out dir under $B
  echo "$B/$1/${2}_c$3"
}

fold_od() { # <target> <rung> <seed> <chunk> <chunks> <chip>  -- mps narrowing 5->2->1 on OOM
  local t=$1 rung=$2 seed=$3 c=$4 k=$5 u=$6 mps s rc secs oom d ob
  ob=$(outbase opendde $t $c)
  for mps in 5 2 1; do
    s=$(date +%s)
    guarded_fold $B/opendde_${t}_c${c}_mps$mps.log $u $PY_SYS -u -m tt_bio.main predict \
      examples/abag_xm/$t.yaml --model opendde-abag --out_dir $ob --override \
      --diffusion_samples $((rung/k)) --max_parallel_samples $mps --seed $seed --host_threads 2 \
      --msa_dir $MSA --msa_cache_only
    rc=$?; secs=$(( $(date +%s) - s ))
    d=$(ls -d $ob/*results_$t 2>/dev/null | head -1)
    read -r _n _di <<<$(count_structs "$d")
    oom=$(grep -c 'Out of Memory' $B/opendde_${t}_c${c}_mps$mps.log 2>/dev/null)
    record opendde-abag $t $rung $seed $c $k $mps $u $rc $secs ${_n:-0} ${_di:-0} ${oom:-0}
    [ "${oom:-0}" -gt 0 ] && [ "${_n:-0}" -eq 0 ] && [ "$mps" != 1 ] || break
  done
}

fold_px() { # <target> <rung> <seed> <chunk> <chunks> <chip>  -- mps 5, narrow 5->1 on OOM
  local t=$1 rung=$2 seed=$3 c=$4 k=$5 u=$6 mps s rc secs oom d ob
  ob=$(outbase protenix $t $c)
  for mps in 5 1; do
    s=$(date +%s)
    guarded_fold $B/protenix_${t}_c${c}_mps$mps.log $u $PY_SYS -u -m tt_bio.main predict \
      examples/abag_xm/$t.yaml --model protenix-v2 --out_dir $ob --override \
      --diffusion_samples $((rung/k)) --max_parallel_samples $mps --seed $seed --host_threads 2 \
      --msa_dir $MSA --msa_cache_only
    rc=$?; secs=$(( $(date +%s) - s ))
    d=$(ls -d $ob/*results_$t 2>/dev/null | head -1)
    read -r _n _di <<<$(count_structs "$d")
    oom=$(grep -c 'Out of Memory' $B/protenix_${t}_c${c}_mps$mps.log 2>/dev/null)
    record protenix-v2 $t $rung $seed $c $k $mps $u $rc $secs ${_n:-0} ${_di:-0} ${oom:-0}
    [ "${oom:-0}" -gt 0 ] && [ "${_n:-0}" -eq 0 ] && [ "$mps" != 1 ] || break
  done
}

fold_esm() { # <target> <rung> <seed> <chunk> <chunks> <chip>  -- single-seq, auto chunking
  local t=$1 rung=$2 seed=$3 c=$4 k=$5 u=$6 s rc secs oom d ob
  ob=$(outbase esmfold2 $t $c)
  s=$(date +%s)
  guarded_fold $B/esmfold2_${t}_c$c.log $u $PY_VENV -u -m tt_bio.main predict \
    examples/abag_xm/$t.yaml --model esmfold2 --out_dir $ob --override \
    --diffusion_samples $((rung/k)) --recycling_steps 10 --sampling_steps 100 --seed $seed \
    --host_threads 2
  rc=$?; secs=$(( $(date +%s) - s ))
  d=$(ls -d $ob/*results_$t 2>/dev/null | head -1)
  read -r _n _di <<<$(count_structs "$d")
  oom=$(grep -c 'Out of Memory' $B/esmfold2_${t}_c$c.log 2>/dev/null)
  record esmfold2 $t $rung $seed $c $k auto $u $rc $secs ${_n:-0} ${_di:-0} ${oom:-0}
}

fold() { # <model> <target> <rung> <seed> <chunk> <chunks> <chip>
  case "$1" in
    opendde-abag) fold_od  "$2" "$3" "$4" "$5" "$6" "$7";;
    protenix-v2)  fold_px  "$2" "$3" "$4" "$5" "$6" "$7";;
    esmfold2)     fold_esm "$2" "$3" "$4" "$5" "$6" "$7";;
    *)            echo "SKIP: $1 not in this window" >> $B/slots.log;;
  esac
}

# Same claim-release fix as p31_fleet.sh, and it matters more here: 48 cells over 27 slots means
# a single never-released claim removes 1/8 of a target's 512 pool and the completeness gate then
# drops that whole (target, 512) rung. See the comment in p31_fleet.sh for the mechanism.
ATTEMPT_MAX=${ATTEMPT_MAX:-3}
slot() {
  local chip=$1 idx n model t rung seed c k tries claimed
  n=$(wc -l < $TASKS)
  while true; do
    claimed=0
    for ((idx=1; idx<=n; idx++)); do
      mkdir $B/claims/$idx 2>/dev/null || continue
      claimed=1
      read -r model t rung seed c k <<<"$(sed -n "${idx}p" $TASKS)"
      ( cd $SRC && export CLAIM=$B/claims/$idx && fold "$model" "$t" "$rung" "$seed" "$c" "$k" "$chip" )
      [ -e $B/claims/$idx/ok ] && continue
      tries=$(( $(cat $B/tries/$idx 2>/dev/null || echo 0) + 1 ))
      echo $tries > $B/tries/$idx
      if [ $tries -lt $ATTEMPT_MAX ]; then
        rm -rf $B/claims/$idx
        echo "$(date -u +%FT%TZ) RELEASE idx=$idx $model $t c$c try=$tries" >> $B/slots.log
      else
        echo "$(date -u +%FT%TZ) EXHAUSTED idx=$idx $model $t c$c after $tries tries" >> $B/slots.log
      fi
    done
    [ "$claimed" = 0 ] && break
  done
  echo "slot $chip done" >> $B/slots.log
}

for c in $CHIPS; do
  slot "$c" &
  sleep "$STAGGER"
done
wait
echo P32_DONE >> $B/results.jsonl

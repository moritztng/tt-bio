#!/bin/bash
# AbAg-XM panel completion, GALAXY window p34 (workstream abag-xm-panel-complete-164):
# the 5 class-G chunk-folds that windows p29-p32 recorded as failures and that were never
# folded, on the frozen p27-era engine tree.
#
# WHY THESE FIVE, AND WHY THE GUARDS CHANGE AND NOTHING ELSE CHANGES
#
# All five died with rc=124. That is the RUNNER's kill code, not the engine's. Two of
# guarded_fold's three kill paths fire on a healthy fold:
#
#   * zero-log kill. p31 ran ZERO_MIN=99, so a fold that prints nothing for 99 minutes
#     dies. It killed opendde 9d73 c4 ten consecutive times, ~99 min burned per attempt.
#   * no-progress kill. STALL_MIN=45 polls in which group CPU grew by less than
#     MIN_CPU_S=60 s AND the log did not grow. With POLL_S=60 that threshold is exactly
#     one core-second per wall-second, so a single-threaded host-side phase (a healthy
#     one gains 59-60 s of CPU per 60 s poll) is scored as a stall or not depending on
#     integer rounding in `ps times=`. A one-thread phase is indistinguishable from a
#     hang by construction.
#
# Two of the five were py-spy'd while live and reported as legitimate compute in
# update_msa, at 15.9 h and 13.2 h, and then killed by the guard. Sibling chunks of the
# same targets complete normally on this same tree at this same mps (od 9d73 c7 in 3542 s,
# od 9xqn c6 in 2462 s), and the failures reproduce across five different chips, so it is
# neither the target nor the chip. The output dirs are empty skeletons with no *.lock, so
# it is not a stale lock either.
#
# So the guard is what changes, not the cell, and not the engine:
#
#   ZERO_MIN=1440   a silent fold gets 24 h, not 99 min
#   STALL_MIN=240   4 h of genuine no-progress before a kill
#   MIN_CPU_S=5     5 core-s per 60 s poll still catches a dead process, and cannot score
#                   a healthy single-threaded phase as a stall
#   CAP_S=86400     two of these were killed while healthy at 15.9 h and 13.2 h
#
# These are runner constants. They have zero engine effect and zero effect on any other
# prediction in the panel.
#
# No retry loop (ATTEMPT_MAX=1): 16 attempts across three of these cells at the old guard
# values proved that a second attempt at the same values buys nothing.
#
# Engine: $H/deepn_src, the frozen p27-era tree every sibling chunk of these five cells was
# folded on, so each cell stays internally single-engine.
#
# Seeds are the campaign ladder, base + 1000*chunk (od 20000, px 30000) -- the value each
# cell's sibling chunks used. Substituting a seed would change the provenance of one chunk
# in 5248.
#
# CHIPS are UMD ids. On this Galaxy UMD ids and /dev/tenstorrent/N node numbers are
# DIFFERENT NAMESPACES and disagree for all 32 chips (UMD ids are assigned in sorted-PCI-BDF
# order; the node numbering is not). UMD 26-30 = /dev nodes 2-6, which are the free chips;
# UMD 2-6 would open nodes 18-22, which the live JapanFold service is folding on right now.
# The inherited guarded_fold mixed the two in one variable -- it launched with
# TT_VISIBLE_DEVICES=$u and quarantined with `tt-smi -r /dev/tenstorrent/$u` using the same
# $u -- so every reset in windows p27-p32 reset a chip other than the one that failed, on a
# box shared with a live customer-facing service. umd_to_node() below converts, derived at
# runtime from /sys rather than hard-coded, because a driver reload can reorder it.
set -u
H=$HOME/mthuening
SRC=$H/deepn_src
B=$H/p34; mkdir -p $B $B/claims $B/tries
STAGGER=${STAGGER:-8}
CHIPS=${CHIPS:-"26 27 28 29 30"}
PY_SYS=/usr/bin/python3.10
MSA=$H/abag_xm/msa_cache

# Honest guards. See the header. Runner-only, no engine effect.
export ZERO_MIN=${ZERO_MIN:-1440}
export STALL_MIN=${STALL_MIN:-240}
export MIN_CPU_S=${MIN_CPU_S:-5}
export CAP_S=${CAP_S:-86400}
export POLL_S=${POLL_S:-60}
ATTEMPT_MAX=${ATTEMPT_MAX:-1}

TASKS=$B/tasks.txt
cat > $TASKS <<'EOF'
opendde-abag 9gvn 512 22000 2 8
opendde-abag 9xqn 512 27000 7 8
opendde-abag 9d73 512 24000 4 8
protenix-v2 9d73 512 30000 0 8
protenix-v2 9ssm 512 37000 7 8
EOF
echo "tasks: $(wc -l < $TASKS)  chips(UMD): [$CHIPS]"

# --- UMD id -> /dev/tenstorrent node, derived at runtime ------------------------------------
# UMD chip ids are assigned in sorted PCI-BDF order. /sys/class/tenstorrent gives node -> BDF.
# Rank the BDFs and the rank IS the UMD id.
declare -A UMD2NODE
build_umd_map() {
  local rank=0 line node
  while read -r _bdf node; do
    UMD2NODE[$rank]=$node
    rank=$((rank + 1))
  done < <(
    for d in /sys/class/tenstorrent/tenstorrent!*; do
      n=$(basename "$d" | sed 's/tenstorrent!//')
      echo "$(basename "$(readlink -f "$d/device")") $n"
    done | sort
  )
  echo "umd->node map: $(for k in $(echo "${!UMD2NODE[@]}" | tr ' ' '\n' | sort -n); do printf '%s=%s ' $k ${UMD2NODE[$k]}; done)" >> $B/slots.log
}
build_umd_map
umd_to_node() { echo "${UMD2NODE[$1]:-$1}"; }

group_cpu() { # <pgid> -> total CPU seconds of every process in the group
  ps -eo pgid=,times= | awk -v g="$1" '$1==g {s+=$2} END {print s+0}'
}

guarded_fold() { # <logfile> <chip-UMD> <cmd...> -- setsid launch + stall/cap group kills + quarantine
  local log=$1 u=$2; shift 2
  local stall_min=${STALL_MIN:-240} cap_s=${CAP_S:-86400} grace_s=${GRACE_S:-120} min_cpu=${MIN_CPU_S:-5}
  local zero_min=${ZERO_MIN:-1440}
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
      echo "$(date -u +%FT%TZ) GUARD: zero-log kill (${zero}m at 0 bytes) -> INT pgid $pid" >> "$log"
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
    local node=$(umd_to_node $u)
    echo "$(date -u +%FT%TZ) GUARD: quarantine UMD $u = /dev/tenstorrent/$node" >> "$log"
    sudo -n tt-smi -r /dev/tenstorrent/$node >> "$log" 2>&1 \
      || echo "$(date -u +%FT%TZ) GUARD: tt-smi reset failed on node $node (UMD $u)" >> "$log"
    sleep 10
  fi
  return $rc
}

record() {  # record <model> <target> <rung> <seed> <chunk> <chunks> <mps> <chip> <rc> <secs> <cifs> <distinct> <oom>
  printf '{"model":"%s","target":"%s","rung":%s,"seed":%s,"chunk":%s,"chunks":%s,"mps":"%s","umd":%s,"rc":%s,"seconds":%s,"cifs":%s,"distinct":%s,"oom":%s}\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" >> $B/results.jsonl
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

outbase() { echo "$B/$1/${2}_c$3"; }

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

fold() { # <model> <target> <rung> <seed> <chunk> <chunks> <chip>
  case "$1" in
    opendde-abag) fold_od "$2" "$3" "$4" "$5" "$6" "$7";;
    protenix-v2)  fold_px "$2" "$3" "$4" "$5" "$6" "$7";;
    *)            echo "SKIP: $1 not in this window" >> $B/slots.log;;
  esac
}

slot() {
  local chip=$1 idx n model t rung seed c k tries claimed
  n=$(wc -l < $TASKS)
  while true; do
    claimed=0
    for ((idx=1; idx<=n; idx++)); do
      mkdir $B/claims/$idx 2>/dev/null || continue
      claimed=1
      read -r model t rung seed c k <<<"$(sed -n "${idx}p" $TASKS)"
      echo "$(date -u +%FT%TZ) CLAIM idx=$idx $model $t c$c on UMD $chip (/dev node $(umd_to_node $chip))" >> $B/slots.log
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
echo P34_DONE >> $B/results.jsonl

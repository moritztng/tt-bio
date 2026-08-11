#!/bin/bash
# AbAg-XM panel completion, GALAXY window p35 (workstream abag-xm-panel-complete-164):
# esmfold2 9j4c, the last of the four "documented WH DRAM exclusions", at N=512.
#
# 9j4c (1095 tokens) refused 627916800 B = 1095x1120x256x2 on a 12 GiB Wormhole chip at the
# campaign's 64 diffusion samples. Root cause: DiffusionConditioningModel.cond_pair held the
# [B,L,L,c_z+c_rel] concat AND its layer_norm live while z_proj allocated its own
# [B,L,L,256], on a chip the trunk had already left ~10 GiB resident on. Row-tiling cond_pair
# over dim=1 (bit-exact -- every op in it is row-local) removes the two full-size transients.
# Verified at the REAL campaign config (loops=10, steps=100, samples=64): FOLDED OK, where
# the unfixed tree on the same chip died at 18 minutes.
#
# Engine: $H/main_src = origin/main 7d83207d5 + the cond_pair fix (branch
# wk/abag-xm-panel-complete-164). No chunk of this cell exists, so all 8 fold on this one
# engine and the cell is internally single-engine, which is the property the campaign's
# per-cell provenance rule actually needs.
#
# Config is the campaign's own esmfold2 protocol: single-sequence (no MSA flags),
# --diffusion_samples 64 --recycling_steps 10 --sampling_steps 100 --host_threads 2, seeds
# 50000 + 1000*chunk. Wormhole auto-enables --fast for esmfold2, as it did for every other
# esmfold2 cell in the panel.
#
# CHIPS are UMD ids. UMD and /dev/tenstorrent/N are different namespaces on this Galaxy and
# disagree for all 32 chips; umd_to_node() converts at runtime from sorted PCI-BDF order so
# a quarantine resets the chip that actually failed. Same guard values as p34: they are
# runner constants with no engine effect.
set -u
H=$HOME/mthuening
SRC=$H/main_src
B=$H/p35; mkdir -p $B $B/claims $B/tries
STAGGER=${STAGGER:-8}
CHIPS=${CHIPS:-"27 28 29 30"}
PY_VENV=$H/tt-bio/env/bin/python3.10
export ZERO_MIN=${ZERO_MIN:-1440}
export STALL_MIN=${STALL_MIN:-240}
export MIN_CPU_S=${MIN_CPU_S:-5}
export CAP_S=${CAP_S:-86400}
export POLL_S=${POLL_S:-60}
ATTEMPT_MAX=${ATTEMPT_MAX:-2}

TASKS=$B/tasks.txt
{ for j in 0 1 2 3 4 5 6 7; do echo "esmfold2 9j4c 512 $((50000 + 1000 * j)) $j 8"; done; } > $TASKS
echo "tasks: $(wc -l < $TASKS)  chips(UMD): [$CHIPS]  engine: $SRC"

declare -A UMD2NODE
build_umd_map() {
  local rank=0 node
  while read -r _bdf node; do UMD2NODE[$rank]=$node; rank=$((rank + 1)); done < <(
    for d in /sys/class/tenstorrent/tenstorrent!*; do
      n=$(basename "$d" | sed 's/tenstorrent!//')
      echo "$(basename "$(readlink -f "$d/device")") $n"
    done | sort
  )
}
build_umd_map
umd_to_node() { echo "${UMD2NODE[$1]:-$1}"; }

group_cpu() { ps -eo pgid=,times= | awk -v g="$1" '$1==g {s+=$2} END {print s+0}'; }

guarded_fold() { # <logfile> <chip-UMD> <cmd...>
  local log=$1 u=$2; shift 2
  local stall_min=${STALL_MIN:-240} cap_s=${CAP_S:-86400} grace_s=${GRACE_S:-120} min_cpu=${MIN_CPU_S:-5}
  local zero_min=${ZERO_MIN:-1440} poll_s=${POLL_S:-60}
  setsid env TT_VISIBLE_DEVICES=$u PYTHONPATH=$SRC "$@" > "$log" 2>&1 &
  local pid=$! t0=$(date +%s) last_cpu=-1 last_size=-1 stall=0 zero=0 killrc=0 g=0
  while kill -0 $pid 2>/dev/null; do
    sleep $poll_s
    kill -0 $pid 2>/dev/null || break
    local cpu=$(group_cpu $pid) size=$(stat -c %s "$log" 2>/dev/null || echo 0)
    if [ "$size" = "0" ]; then zero=$((zero+1)); else zero=0; fi
    if [ "$last_cpu" -ge 0 ]; then
      if [ $((cpu - last_cpu)) -ge $min_cpu ] || [ "$size" != "$last_size" ]; then stall=0; else stall=$((stall+1)); fi
    fi
    last_cpu=$cpu; last_size=$size
    if [ $zero -ge $zero_min ] || [ $stall -ge $stall_min ]; then
      echo "$(date -u +%FT%TZ) GUARD: kill (zero=${zero}m stall=${stall}m) -> INT pgid $pid" >> "$log"
      kill -INT -- -$pid 2>/dev/null
      g=0; while kill -0 $pid 2>/dev/null && [ $g -lt $grace_s ]; do sleep 2; g=$((g+2)); done
      kill -0 $pid 2>/dev/null && kill -KILL -- -$pid 2>/dev/null
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
    sudo -n tt-smi -r /dev/tenstorrent/$node >> "$log" 2>&1 || true
    sleep 10
  fi
  return $rc
}

record() {
  printf '{"model":"%s","target":"%s","rung":%s,"seed":%s,"chunk":%s,"chunks":%s,"mps":"%s","umd":%s,"rc":%s,"seconds":%s,"cifs":%s,"distinct":%s,"oom":%s}\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" >> $B/results.jsonl
  if [ -n "${CLAIM:-}" ] && [ "$9" = 0 ] && [ "${11}" -eq $(( $3 / $6 )) ] && [ "${12}" -eq "${11}" ]; then touch "$CLAIM/ok"; fi
}

count_structs() {
  local d=$1 n distinct
  n=$(ls $d/structures/*.cif 2>/dev/null | wc -l)
  distinct=$(md5sum $d/structures/*.cif 2>/dev/null | awk '{print $1}' | sort -u | wc -l)
  echo "${n:-0} ${distinct:-0}"
}

fold_esm() { # <target> <rung> <seed> <chunk> <chunks> <chip>
  local t=$1 rung=$2 seed=$3 c=$4 k=$5 u=$6 s rc secs oom d ob
  ob=$B/esmfold2/${t}_c$c
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
      ( cd $SRC && export CLAIM=$B/claims/$idx && fold_esm "$t" "$rung" "$seed" "$c" "$k" "$chip" )
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

for c in $CHIPS; do slot "$c" & sleep "$STAGGER"; done
wait
echo P35_DONE >> $B/results.jsonl

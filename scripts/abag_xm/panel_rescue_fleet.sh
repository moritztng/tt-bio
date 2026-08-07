#!/bin/bash
# AbAg-XM panel rescue fleet (workstream abag-xm-panel-to-100pct).
#
# Re-folds the five Class-C1 chunks that hit the raw 21600s cap in p27 -- they are
# seed-deterministically slow (py-spy: legitimate trunk compute in swiglu/_msa, ~100%
# CPU, three different chips for 9gvn c2), NOT hung and NOT chip-bound. Runs them with
# the fixed guard: no ZERO_MIN leg (a redirected tt-bio predict writes nothing until
# completion, so 0-byte carries no information), CAP_S=57600 (16h, ~25x the sibling
# median), the two-leg no-progress rule unchanged.
#
# Default task list (rung=256, k=4 -> --diffusion_samples 64; records in p27's schema):
#   boltz2        9ua5 c2  seed 42000  mps 1
#   opendde-abag  9gvn c2  seed 22000  mps 5->2->1 on OOM
#   opendde-abag  9rye c2  seed 22000  mps 5->2->1 on OOM
#   opendde-abag  9rye c3  seed 23000  mps 5->2->1 on OOM
#   opendde-abag  9xqn c2  seed 22000  mps 5->2->1 on OOM
#
# B (window dir) and RESCUE_TASKS are env-overridable so the same script can carry a
# p29 9d73 re-fold (B=$H/p29) if one caps at 21600 there. A task that already has an
# rc=0 record in $B/results.jsonl is skipped; a task that hits CAP_S is a named,
# evidenced permanent exclusion (record its py-spy stack before ending the pass).
#
# CHIP COURTESY: pass CHIPS="0 1 2 3 4" (exactly one chip per task). Do NOT launch
# while another fleet holds the chip set. Never edit tasks.txt; claims live in
# $B/rescue_claims so the original windows' claim namespaces are untouched.
set -u
H=$HOME/mthuening
SRC=$H/deepn_src
B=${B:-$H/p27}; mkdir -p $B $B/rescue_claims
PY_SYS=/usr/bin/python3.10
PY_VENV=$H/tt-bio/env/bin/python3.10
MSA=$H/abag_xm/msa_cache
STAGGER=8

TASKS=$B/rescue_tasks.txt
if [ -n "${RESCUE_TASKS:-}" ]; then
  printf '%s\n' $RESCUE_TASKS > $TASKS
elif [ ! -s $TASKS ]; then
  cat > $TASKS <<'EOF'
boltz2 9ua5 256 42000 2 4
opendde-abag 9gvn 256 22000 2 4
opendde-abag 9rye 256 22000 2 4
opendde-abag 9rye 256 23000 3 4
opendde-abag 9xqn 256 22000 2 4
EOF
fi
NTask=$(wc -l < $TASKS)
CHIPS=${CHIPS:-$(seq -s' ' 0 $((NTask-1)))}
[ -n "${CHIPS// /}" ] || { echo "CHIPS is empty -- nothing to launch"; exit 0; }
[ -d "$SRC" ] || { echo "engine tree $SRC missing -- refusing to launch"; exit 1; }
command -v "$PY_SYS" >/dev/null || { echo "$PY_SYS missing -- refusing to launch"; exit 1; }
echo "rescue tasks: $NTask  chips: $(wc -w <<<"$CHIPS") [$CHIPS]  window: $B"

group_cpu() { # <pgid> -> total CPU seconds of every process in the group
  ps -eo pgid=,times= | awk -v g="$1" '$1==g {s+=$2} END {print s+0}'
}

guarded_fold() { # <logfile> <chip> <cmd...> -- setsid launch + stall/cap group kills + quarantine
  local log=$1 u=$2; shift 2
  # No ZERO_MIN leg: a redirected tt-bio predict emits nothing until completion, so a
  # 0-byte log is not a hang signal. CAP_S default 57600 (16h) — the slowest legitimate
  # fold on record needed >9.5x the old 21600 cap; 16h is ~25x the sibling median.
  local stall_min=${STALL_MIN:-45} cap_s=${CAP_S:-57600} grace_s=${GRACE_S:-120} min_cpu=${MIN_CPU_S:-60}
  local poll_s=${POLL_S:-60}
  setsid env TT_VISIBLE_DEVICES=$u "$@" > "$log" 2>&1 &
  local pid=$! t0=$(date +%s) last_cpu=-1 last_size=-1 stall=0 killrc=0 g=0
  while kill -0 $pid 2>/dev/null; do
    sleep $poll_s
    kill -0 $pid 2>/dev/null || break
    local cpu=$(group_cpu $pid) size=$(stat -c %s "$log" 2>/dev/null || echo 0)
    if [ "$last_cpu" -ge 0 ]; then
      if [ $((cpu - last_cpu)) -ge $min_cpu ] || [ "$size" != "$last_size" ]; then
        stall=0
      else
        stall=$((stall+1))
      fi
    fi
    last_cpu=$cpu; last_size=$size
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
    # NO unattended tt-smi here: this fleet runs ALONGSIDE prod on the same Galaxy,
    # UMD id != /dev node number (verified 2026-08-07: UMD 8/16/29 -> nodes 24/8/5),
    # and tt-smi can escalate a targeted reset to -glx_reset (all 32 chips). A wrong
    # or escalated reset kills live prod workers. Mark the chip suspect instead; a
    # later pass resets it deliberately. A dirty chip fails the next fold fast at
    # device open and is recorded, not silent.
    echo "$(date -u +%FT%TZ) GUARD: chip umd $u SUSPECT after kill -- needs a deliberate reset" >> "$log"
    echo "$u $(date -u +%FT%TZ)" >> $B/rescue_suspect_chips.txt
    sleep 10
  fi
  return $rc
}

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

already_ok() { # <model> <target> <rung> <seed> <chunk> -- true if an rc=0 record exists
  local m=$1 t=$2 rung=$3 seed=$4 c=$5
  [ "$($PY_SYS - "$B/results.jsonl" "$m" "$t" "$rung" "$seed" "$c" <<'PY'
import json, sys
path, m, t, rung, seed, c = sys.argv[1:7]
n = 0
try:
    for line in open(path):
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if (r.get("model"), r.get("target"), str(r.get("rung")), str(r.get("seed")),
                str(r.get("chunk"))) == (m, t, rung, seed, c) \
                and r.get("rc") in (0, "0") and r.get("cifs", 0) > 0:
            n += 1
except FileNotFoundError:
    pass
print(n)
PY
)" -gt 0 ]
}

fold_bz() { # <target> <rung> <seed> <chunk> <chunks> <chip>  -- mps 1 (p27 convention)
  local t=$1 rung=$2 seed=$3 c=$4 k=$5 u=$6 s rc secs oom d ob
  ob=$(outbase boltz2 $t $c $k)
  s=$(date +%s)
  guarded_fold $B/boltz2_${t}_c$c.log $u $PY_SYS -u -m tt_bio.main predict \
    examples/abag_xm/$t.yaml --model boltz2 --out_dir $ob --override \
    --diffusion_samples $((rung/k)) --max_parallel_samples 1 --seed $seed --host_threads 2 \
    --msa_dir $MSA --msa_cache_only
  rc=$?; secs=$(( $(date +%s) - s ))
  d=$(ls -d $ob/*results_$t 2>/dev/null | head -1)
  read -r _n _di <<<$(count_structs "$d")
  oom=$(grep -c 'Out of Memory' $B/boltz2_${t}_c$c.log 2>/dev/null)
  record boltz2 $t $rung $seed $c $k 1 $u $rc $secs ${_n:-0} ${_di:-0} ${oom:-0}
}

fold_od() { # <target> <rung> <seed> <chunk> <chunks> <chip>  -- mps narrowing 5->2->1 on OOM
  local t=$1 rung=$2 seed=$3 c=$4 k=$5 u=$6 mps s rc secs oom d ob
  ob=$(outbase opendde $t $c $k)
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
  ob=$(outbase protenix $t $c $k)
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
  ob=$(outbase esmfold2 $t $c $k)
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
    boltz2)       fold_bz  "$2" "$3" "$4" "$5" "$6" "$7";;
    opendde-abag) fold_od  "$2" "$3" "$4" "$5" "$6" "$7";;
    protenix-v2)  fold_px  "$2" "$3" "$4" "$5" "$6" "$7";;
    esmfold2)     fold_esm "$2" "$3" "$4" "$5" "$6" "$7";;
    *)            echo "SKIP: $1 not supported" >> $B/rescue_slots.log;;
  esac
}

slot() {
  local chip=$1 idx n model t rung seed c k
  n=$(wc -l < $TASKS)
  for ((idx=1; idx<=n; idx++)); do
    mkdir $B/rescue_claims/$idx 2>/dev/null || continue
    read -r model t rung seed c k <<<"$(sed -n "${idx}p" $TASKS)"
    if already_ok "$model" "$t" "$rung" "$seed" "$c"; then
      echo "rescue slot $chip: $model/$t c$c already rc=0 -- skip" >> $B/rescue_slots.log
      continue
    fi
    if [ -f $B/rescue_suspect_chips.txt ] && grep -q "^$chip " $B/rescue_suspect_chips.txt; then
      echo "rescue slot $chip: chip marked suspect -- skipping $model/$t c$c (needs reset first)" >> $B/rescue_slots.log
      continue
    fi
    ( cd $SRC && fold "$model" "$t" "$rung" "$seed" "$c" "$k" "$chip" )
  done
  echo "rescue slot $chip done" >> $B/rescue_slots.log
}

for c in $CHIPS; do
  slot "$c" &
  sleep "$STAGGER"
done
wait
echo RESCUE_DONE >> $B/rescue_slots.log

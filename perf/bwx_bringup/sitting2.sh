#!/bin/bash
# The redo sitting: everything bwx-bringup still owes, on a chip whose canary passes, under ONE
# agent restart. Card 30 of dev .107 went bad mid-pass on 2026-09-29 and the box was recovered by
# hand at 20:28Z; every result below is taken after that, so it rests on a chip known good at the
# time rather than on the chip that was silently miscomputing.
#
# Ordered so that being cut off early still lands the load-bearing answers:
#   1 grade_re   the float64 VJP grade, 4 blocks + both controls  -- charter item 3, and the
#                gate on bwx-perf believing any earlier number
#   2 grade_45   the EXACT configuration that failed twice on the bad chip (non-finite once,
#                Bus error / Non-existent physical address once). This is the discriminator:
#                if it grades clean here, the fault was that chip, not a Wormhole software
#                difference reachable from the AF2 backward.
#   3 round_re   the (1,1,hifi) round, so 2.18x rests on the good chip too
#   4 ceiling    the token ladder -- charter item 4, the one item with no measurement at all
#
# Each stage writes out/<stage>/ and a DONE-<stage> marker, so a relaunch can tell what finished
# without re-reading logs. The float64 prologue peaks near 353 GB RSS on this host, so every
# afgrad stage carries memguard on its own pid only.
set -uo pipefail
cd ~/bwx
OUT=${BWX_OUT:-~/bwx/out/sitting2}
CHIP=${BWX_CHIP:-30}
PARAMS=$HOME/bwx/af2_params/params_model_1_ptm.npz
PY=~/bwx/venv/bin/python
mkdir -p "$OUT"
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/sitting.log"; }

say "sitting2 start chip=$CHIP load=$(cat /proc/loadavg | cut -d' ' -f1-3)"

# --- stages 1 and 2: the float64 gradient grade -------------------------------------------
grade(){  # grade <stage> <blocks> <tag> [extra args...]
  stage=$1; blocks=$2; tag=$3; shift 3
  d="$OUT/$stage"; mkdir -p "$d"
  say "$stage start blocks=$blocks tag=$tag"
  $PY -u ~/bwx/tt-bio/perf/bcx_afgrad/afgrad.py vjp --n 288 --blocks "$blocks" \
      --tag "$tag" --params "$PARAMS" "$@" > "$d/grade.log" 2>&1 &
  gp=$!
  bash ~/bwx/memguard.sh "$gp" 25 >> "$d/memguard.log" 2>&1 &
  wait $gp; rc=$?
  say "$stage rc=$rc"
  # The whole point of this stage is that a non-finite reading is a RESULT, not a crash, so
  # look for one explicitly rather than trusting the exit code.
  # \binf\b / \bnan\b, not a bare substring: "inf" matches "info" and every INFO line in a ttnn
  # log, which buries the one reading this stage exists to catch.
  grep -iE "\b(inf|-inf|nan)\b|Bus error|Non-existent physical|Signal:" "$d/grade.log" | head -8 \
      | sed "s/^/  $stage: /" | tee -a "$OUT/sitting.log"
  tail -20 "$d/grade.log" | tee -a "$OUT/sitting.log"
  touch "$OUT/DONE-$stage"
}

grade grade_re "4,5,6,7" wh_chars2 --controls-all
grade grade_45 "4,5"     wh_hifi2

# --- stage 3: the round, same arm as the Blackhole peer ------------------------------------
d="$OUT/round_re"; mkdir -p "$d"
say "round_re start binder=146 arm=(1,1,hifi)"
timeout 1800 $PY -u ~/bwx/tt-bio/perf/bcx_round/run_round.py \
    --rounds 4 --binder 146 --params $HOME/bwx/af2_params --out "$d" \
    --exact 0 --triatt-hifi 1 --rne-kernel 1 --triatt-bw 1 --extra-msa 1 --template 1 \
    > "$d/round.log" 2>&1
say "round_re rc=$?"
grep -E "^round |median|s a round" "$d/round.log" | tail -6 | tee -a "$OUT/sitting.log"
touch "$OUT/DONE-round_re"

# --- stage 4: the token ceiling ------------------------------------------------------------
say "ceiling start"
BWX_OUT="$OUT/ceiling" bash ~/bwx/ceiling.sh
touch "$OUT/DONE-ceiling"

say "sitting2 done"

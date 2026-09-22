#!/bin/bash
# of3t-stepfloor: the timed arms, run in sequence because one card takes one device context.
# Rooted in THIS row's worktree so fleet hygiene cannot delete the cwd out from under it.
cd /home/ttuser/.coworker/wt/of3t-stepfloor || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python
L=/tmp/of3t/stepfloor
O=perf/of3t_stepfloor/out
mkdir -p "$L" "$O"
run () {  # run <tag> <extra env> <args...>
  tag=$1; shift
  echo "=== $tag start $(date -u +%FT%TZ) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> "$L/arms.log"
  env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:of3t-stepfloor \
      "$@" > "$L/$tag.log" 2>&1
  echo "=== $tag done  $(date -u +%FT%TZ) rc=$?" >> "$L/arms.log"
}
# 1. the charter: the taped step, 3 reps, shipped default (RENORM ON in wk/of3t)
run step_taped $PY perf/of3t_stepfloor/fullstep.py --tokens 384 --cycles 1 --samples 4 \
    --reps 3 --out $O/step_taped_384.json
# 2. D32's ratio at STEP scope, same process shape
run step_untaped $PY perf/of3t_stepfloor/fullstep.py --tokens 384 --cycles 1 --samples 4 \
    --reps 3 --no-tape --no-optimizer --out $O/step_untaped_384.json
# 3. D56: the same step with the lever OFF
run step_renorm_off env TT_BIO_SOFTMAX_BW_RENORM=0 $PY perf/of3t_stepfloor/fullstep.py \
    --tokens 384 --cycles 1 --samples 4 --reps 2 --out $O/step_renormoff_384.json
# 4. D164's residual: the pass-54 rung again with the lever OFF, probe still inert
run d164_renorm_off env TT_BIO_SOFTMAX_BW_RENORM=0 $PY perf/of3t_l1/ladder.py --tokens 384 \
    --backward --probe-every 0 --out $O/d164_probeoff_renormoff_384.json
echo "ALL ARMS DONE $(date -u +%FT%TZ)" >> "$L/arms.log"

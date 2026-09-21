#!/bin/bash
# of3t-stepfloor pass 2: the STEADY step, and the lever priced inside ONE warm process.
# Rooted in this row's own worktree so fleet hygiene cannot delete the cwd under it.
cd /home/ttuser/.coworker/wt/of3t-stepfloor || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python
L=/tmp/of3t/stepfloor
O=perf/of3t_stepfloor/out
mkdir -p "$L" "$O"
# One card takes one device context: wait out pass 1's chain before opening dev3.
while pgrep -f "perf/of3t_l1/ladder.py --tokens 384 --backward --probe-every 0" >/dev/null; do
  sleep 10
done
run () {
  tag=$1; shift
  echo "=== $tag start $(date -u +%FT%TZ) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> "$L/arms2.log"
  env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:of3t-stepfloor \
      "$@" > "$L/$tag.log" 2>&1
  echo "=== $tag done  $(date -u +%FT%TZ) rc=$?" >> "$L/arms2.log"
}
# A. cold ON, steady ON, steady OFF, steady ON. rep0 vs pass 1's 585.062 s is the
#    cross-process cold A/A; reps 1 and 3 are the spread; rep 2 is the lever, paired.
run step_rekey_a $PY perf/of3t_stepfloor/fullstep.py --tokens 384 --cycles 1 --samples 4 \
    --reps 4 --renorm-per-rep 1,1,0,1 --out $O/step_rekey_384.json
# B. the mirror: cold OFF against pass 1's 685.579 s, then OFF / ON / OFF.
run step_rekey_b $PY perf/of3t_stepfloor/fullstep.py --tokens 384 --cycles 1 --samples 4 \
    --reps 4 --renorm-per-rep 0,0,1,0 --out $O/step_rekey_b_384.json
echo "ALL ARMS2 DONE $(date -u +%FT%TZ)" >> "$L/arms2.log"

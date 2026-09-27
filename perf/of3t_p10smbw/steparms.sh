#!/usr/bin/env bash
# of3t-p10smbw: the step-time A/B for the taped-softmax precise forward, BOTH arms on one card.
# The board's 62.237 s was measured on qb1 card 1 (p150a); this runs both arms on qb2 card 1 so
# the ratio is a lever reading and not a host reading.
W=/home/ttuser/.coworker/wt/of3t-p10smbw
cd "$W" || exit 1
L=/tmp/of3t/of3t-p10smbw
mkdir -p "$L" "$W/perf/of3t_p10smbw/out"
BASE_SHA=868a2eaab   # main: _v_softmax with no compute_kernel_config

run () {  # run <tag>
  local tag=$1
  echo "=== $tag start $(date -u +%FT%TZ) head=$(git rev-parse --short HEAD) loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
  bash perf/of3t_p10smbw/arm.sh "step_$tag" \
    /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_stepfloor/fullstep.py \
      --tokens 384 --cycles 4 --samples 48 --chunk 4 --reps 2 --no-exact \
      --out "$W/perf/of3t_p10smbw/out/step_$tag.json" > "$L/step_$tag.log" 2>&1
  echo "=== $tag exit $? $(date -u +%FT%TZ)"
  tail -25 "$L/step_$tag.log"
}

run fix
git checkout $BASE_SHA -- tt_bio/taped_ttnn.py || exit 1
echo "=== reverted _v_softmax to $BASE_SHA: $(git diff --stat)"
run base
git checkout HEAD -- tt_bio/taped_ttnn.py
echo "=== restored: $(git status --porcelain tt_bio/taped_ttnn.py | wc -l) dirty"
echo "=== CHAIN DONE $(date -u +%FT%TZ)"

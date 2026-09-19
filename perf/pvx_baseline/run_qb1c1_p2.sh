#!/bin/bash
# pvx-baseline phase 2 on qb1 card 1 (Blackhole p150a), reordered after pass 2 read the queue.
#
# Pass 1 ran Boltz-2 first and left Protenix behind three sessions. That is the wrong order:
# LEDGER O1 (Protenix 512 aa at a pinned DURING-sampled clock) blocks ALL scoring in this
# campaign and is entirely unmeasured, while the two governor sessions only refine a clock
# term that two prior rows already bracket at 1.267x-1.410x. So Protenix and the two block
# censuses run before anything on the governor.
#
# Waits for the phase-1 b2_old_s2 session to finish first -- it is orphaned but live, and two
# pvx-baseline benchlock waiters would otherwise queue against each other.
set -u
WT=/home/ttuser/.coworker/wt/pvx-baseline
OLD=/home/ttuser/pvx_oldtree_0d69dc1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/pvx_baseline/out
BL=/home/ttuser/.coworker/scripts/benchlock.sh
mkdir -p "$OUT"

# wait out the orphaned phase-1 session
for _ in $(seq 1 240); do
  pgrep -f "cell.py .*--tag b2_old_s2" >/dev/null 2>&1 || break
  sleep 15
done
echo "=== $(date -u +%H:%M:%SZ) phase 1 b2_old_s2 no longer running, phase 2 starts ==="

run() {   # run <tree> <tag> <model> <reps> <clock> <extra...>
  local tree=$1 tag=$2 model=$3 reps=$4 clk=$5; shift 5
  if [ -s "$OUT/$tag.json" ] && grep -q "\"summary\"" "$OUT/$tag.json" 2>/dev/null; then
    echo "=== $(date -u +%H:%M:%SZ) $tag already has a summary, skipping ==="
    return 0
  fi
  echo "=== $(date -u +%H:%M:%SZ) $tag  tree=$tree model=$model reps=$reps clock=$clk $* ==="
  BENCHLOCK_WAIT_S=5400 BENCHLOCK_LOAD_WAIT_S=1800 "$BL" pvx-baseline -- \
    env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:pvx-baseline \
        PYTHONPATH="$tree" \
    "$PY" -u "$tree/perf/pvx_baseline/cell.py" --model "$model" --reps "$reps" --clock "$clk" \
      --out "$OUT/$tag.json" --tag "$tag" "$@"
  echo "RC=$? $tag"
}

run "$WT" ptx_new_s1 protenix-v2 4 1350      # O1: blocks all scoring
run "$WT" ptx_blocks protenix-v2 2 1350 --timers
run "$WT" b2_blocks  boltz2      2 1350 --timers
run "$WT" ptx_new_s2 protenix-v2 4 1350      # session-level A/A floor
run "$OLD" b2_old_gov boltz2 4 0             # clock half of the cited 1.63x
run "$WT"  b2_new_gov boltz2 4 0
run "$WT"  ptx_new_gov protenix-v2 3 0

echo "PVXBASELINEP2DONE $(date -u +%FT%TZ)"

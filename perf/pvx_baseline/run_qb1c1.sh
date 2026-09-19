#!/bin/bash
# The pvx-baseline cell campaign on qb1 card 1 (Blackhole p150a, four single-ASIC boards, so
# there is no board partner sharing a power budget to state), every fold at a pinned 1350 MHz
# sampled DURING the fold at 4 Hz.
#
# Sessions alternate the two Boltz-2 trees rather than interleaving arms inside one process,
# because the two arms ARE two trees -- main today against 0d69dc1de, the merge the 23.504 s
# cell was published from -- and a tree cannot be swapped mid-process. The session is the
# independent unit, so old and new alternate and each contributes two pinned sessions.
#
# The two `gov` sessions run the SAME two trees on the governor. They are what separates the
# cited 1.63x into optimization and clock: the published cell predates clock pinning, so part
# of the speedup against it may be nothing but a hotter chip.
set -u
WT=/home/ttuser/.coworker/wt/pvx-baseline
OLD=/home/ttuser/pvx_oldtree_0d69dc1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/pvx_baseline/out
BL=/home/ttuser/.coworker/scripts/benchlock.sh
mkdir -p "$OUT"

run() {   # run <tree> <tag> <model> <reps> <clock> <extra...>
  local tree=$1 tag=$2 model=$3 reps=$4 clk=$5; shift 5
  if [ -s "$OUT/$tag.json" ] && grep -q '"summary"' "$OUT/$tag.json" 2>/dev/null; then
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

# Boltz-2 pinned, four sessions, old and new alternating.
run "$WT"  b2_new_s1 boltz2 4 1350
run "$OLD" b2_old_s1 boltz2 4 1350
run "$WT"  b2_new_s2 boltz2 4 1350
run "$OLD" b2_old_s2 boltz2 4 1350

# The same two trees on the governor: the clock half of the cited ratio.
run "$OLD" b2_old_gov boltz2 4 0
run "$WT"  b2_new_gov boltz2 4 0

# Protenix v2 today. Two pinned sessions give a session-level A/A floor as well as the
# within-session one.
run "$WT" ptx_new_s1 protenix-v2 4 1350
run "$WT" ptx_new_s2 protenix-v2 4 1350

# The block census. Timers cost two device syncs per call, so these folds are slower than the
# cell by construction and their fold_s is never the cell.
run "$WT" ptx_blocks protenix-v2 2 1350 --timers
run "$WT" b2_blocks  boltz2      2 1350 --timers

# Protenix on the governor, for the same clock decomposition Boltz-2 gets.
run "$WT" ptx_new_gov protenix-v2 3 0

echo "PVXBASELINEDONE $(date -u +%FT%TZ)"

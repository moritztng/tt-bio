#!/bin/bash
# The publishable p300c pair, on card 3 (this worker's grant), pinned at 1350 MHz.
#
# The 23.504 s Boltz-2 cell and the 54.760 s Protenix-v2 cell were BOTH taken on a qb2 p300c,
# so the ratio Moritz may publish has to come from this board class. The qb1 p150a pair reads
# 1.567x pooled over 8+8 folds; this pair says whether that transfers to the p300c.
#
# Card 3 board partner is card 2, on one power budget. cell.py samples the partner per fold,
# and benchlock holds the whole box, so a partner going busy mid-fold is in that fold's record.
set -u
NEW=/home/ttuser/pvx_qb2/new
OLD=/home/ttuser/pvx_qb2/old
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/pvx_qb2/out3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
mkdir -p "$OUT"

run() {   # run <tree> <tag> <model> <reps> <clock> <extra...>
  local tree=$1 tag=$2 model=$3 reps=$4 clk=$5; shift 5
  if [ -s "$OUT/$tag.json" ] && grep -q "\"summary\"" "$OUT/$tag.json" 2>/dev/null; then
    echo "=== $(date -u +%H:%M:%SZ) $tag already has a summary, skipping ==="
    return 0
  fi
  echo "=== $(date -u +%H:%M:%SZ) $tag  tree=$tree model=$model reps=$reps clock=$clk $* ==="
  BENCHLOCK_WAIT_S=25200 BENCHLOCK_LOAD_WAIT_S=1800 "$BL" pvx-baseline -- \
    env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:pvx-baseline \
        PYTHONPATH="$tree" \
    "$PY" -u "$tree/perf/pvx_baseline/cell.py" --model "$model" --reps "$reps" --clock "$clk" \
      --out "$OUT/$tag.json" --tag "$tag" "$@"
  echo "RC=$? $tag"
}

# Old and new alternate session by session: the two arms are two trees and a tree cannot be
# swapped mid-process, so per perf-ab-session-is-the-independent-unit the session is the unit.
run "$NEW" b2_new_s1 boltz2 4 1350
run "$OLD" b2_old_s1 boltz2 4 1350
run "$NEW" b2_new_s2 boltz2 4 1350
run "$OLD" b2_old_s2 boltz2 4 1350

# Protenix v2 today on the board class its 54.760 s anchor came from.
run "$NEW" ptx_new_s1 protenix-v2 4 1350
run "$NEW" ptx_new_s2 protenix-v2 4 1350

echo "PVXBASELINEDONE $(date -u +%FT%TZ)"

#!/bin/bash
# pvx-didittransfer: the four-arm did-it-transfer cell, qb1 card 1 (Blackhole p150a, four
# single-ASIC boards, so no board partner shares the power budget), every fold at a pinned
# 1350 MHz sampled DURING at 4 Hz. Instrument is pvx-baseline's cell.py verbatim (md5
# 21e0770f080a4d965203b59a193107be), so this row's arms and that row's measure the same region.
#
# A tree cannot be swapped mid-process, so arms alternate by session, old and new in turn.
set -u
OLDP=/home/ttuser/pvx_didit/old_ptx        # 61d05cc5 -- the tree the 50.543 s Protenix cell shipped from
OLDB=/home/ttuser/pvx_oldtree_0d69dc1      # 0d69dc1de -- the tree the 23.504 s Boltz-2 cell shipped from
NEW=/home/ttuser/pvx_didit/new             # ec4b66412 -- origin/main today
FIX=$NEW/perf/size512/fixtures             # one fixture pair for every arm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/pvx_didit/out
BL=/home/ttuser/.coworker/scripts/benchlock.sh
mkdir -p "$OUT"

run() {   # run <tree> <cellpath> <tag> <model> <reps> [extra...]
  local tree=$1 cell=$2 tag=$3 model=$4 reps=$5; shift 5
  if [ -s "$OUT/$tag.json" ] && grep -q '"summary"' "$OUT/$tag.json" 2>/dev/null; then
    echo "=== $(date -u +%H:%M:%SZ) $tag already summarised, skipping ==="; return 0
  fi
  echo "=== $(date -u +%H:%M:%SZ) $tag tree=$tree model=$model reps=$reps $* ==="
  BENCHLOCK_WAIT_S=1200 BENCHLOCK_LOAD_WAIT_S=600 "$BL" pvx-didittransfer -- \
    env ${EXTRAENV:-} TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:pvx-didittransfer \
        PYTHONPATH="$tree" \
    "$PY" -u "$tree/$cell" --model "$model" --reps "$reps" --clock 1350 --fixdir "$FIX" \
      --out "$OUT/$tag.json" --tag "$tag" "$@"
  echo "RC=$? $tag"
}

ptx_old() { run "$OLDP" perf/pvx_didit/cell.py "ptx_old_$1" protenix-v2 "${2:-3}"; }
ptx_new() { run "$NEW"  perf/pvx_didit/cell.py "ptx_new_$1" protenix-v2 "${2:-3}"; }
b2_old()  { run "$OLDB" perf/pvx_baseline/cell.py "b2_old_$1" boltz2 "${2:-3}"; }
b2_new()  { run "$NEW"  perf/pvx_didit/cell.py "b2_new_$1" boltz2 "${2:-3}"; }


# The 78ed5a1e attribution arm. NOT a proposal to turn the accurate softmax off: it asks how much
# of this window's Protenix time is the correctness fix that landed inside it. -protenix.trunk and
# -protenix.confidence are the only two accurate_softmax_site tokens protenix.py builds for
# protenix-v2, so this puts today's tree in its pre-78ed5a1e softmax state and changes nothing else.
ptx_nosm() { EXTRAENV="TT_BIO_ACCURATE_SOFTMAX_AB=-protenix.trunk,-protenix.confidence" \
             run "$NEW" perf/pvx_didit/cell.py "ptx_nosm_$1" protenix-v2 "${2:-3}"; }

"$@"

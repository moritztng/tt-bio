#!/usr/bin/env bash
# The quiet re-capture. One 512 aa attrib fold at the head the floor was taken at, on a quiet box,
# with every call's own wall kept.
#
# Why a detached f072ae02f and not the tip: tt_bio/tenstorrent.py alone moved 1169 lines between
# that head and the tip, so a tip fold would change the op set as well as the times and the
# comparison against the 15.031 s floor would confound the two. Only baseline_attrib.py is taken
# from this branch, because that is the change under test.
#
# Writes outside the worktree, so a checkout switch cannot eat a running capture.
set -eu
WT=/home/ttuser/.coworker/wt/roof-quiet-attrib-refold
OUT=${OUT:-/home/ttuser/.coworker/artifacts/roof-quiet-attrib-refold}
HEAD_AT=f072ae02f6de655e21a4d8f388522ae826cc4ca7
CARD=${CARD:-2}
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BRANCH=wk/roof-quiet-attrib-refold

mkdir -p "$OUT/captures" "$OUT/cifs"
cd "$WT"
restore() { git checkout -f "$BRANCH" >/dev/null 2>&1 || true; }
trap restore EXIT

git checkout --detach "$HEAD_AT" >/dev/null 2>&1
git checkout "$BRANCH" -- perf/b2x-baseline-attrib/baseline_attrib.py
git rev-parse HEAD | tee "$OUT/head.txt"

uptime | tee "$OUT/loadavg_before.txt"
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
TT_BIO_LEASE_HOLDER=worker:roof-quiet-attrib-refold \
  "$PY" -u perf/b2x-baseline-attrib/baseline_attrib.py \
    --phases baseline,attrib --reps 2 --size 512 \
    --out "$OUT/attrib_quiet_512_qb2c${CARD}.json" \
    --capdir "$OUT/captures" --cifdir "$OUT/cifs" 2>&1 | tee "$OUT/refold.log"
uptime | tee "$OUT/loadavg_after.txt"

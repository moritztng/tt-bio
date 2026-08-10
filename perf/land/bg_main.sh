#!/bin/bash
# The BoltzGen L1 throw was seen on arm L0 = 83499742, which was main when this leg's arms were cut.
# main is now cc39a867, 60 commits later, and two of those commits are E8's L1-budget correction
# (fa0bfc21, 2135cf01) which moves exactly the gates involved. So "main today crashes BoltzGen" is
# only a live claim if it reproduces on cc39a867. Two runs, fresh dirs; the throw lands at ~19 s.
set -u
WT=/home/ttuser/.coworker/wt/perfwar-qb1-rebaseline-and-land
export TT_VISIBLE_DEVICES=3
export TT_BIO_LEASE_HOLDER=worker:perfwar-qb1-rebaseline-and-land
export TT_BIO_TRIMUL_OUT_FUSED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/usr/bin/python3
LOG="$WT/perf/land/out/w5_bg_repro.log"
OUTROOT=/home/ttuser/w5_xm

while ! grep -aq "W5 BG REPRO COMPLETE" "$LOG" 2>/dev/null; do sleep 15; done

run () {  # label arm cap
  local od="$OUTROOT/$1"
  rm -rf "$od"; mkdir -p "$od"
  echo "=== $(date -u +%H:%M:%S) $1 (arm $2 = $(git -C "$WT/arms/$2" rev-parse --short HEAD), cap $3 s)" >>"$LOG"
  ( cd "$WT/arms/$2" && PYTHONPATH=$PWD timeout "$3" "$PY" -m tt_bio.main design examples/binder.yaml \
      --model boltzgen --seed 0 --out_dir "$od" >>"$LOG" 2>&1 )
  local rc=$?
  local thrown="no"
  grep -aq "TT_THROW" "$od/run.log" 2>/dev/null && thrown="YES"
  echo "=== $(date -u +%H:%M:%S) $1 rc=$rc tt_throw=$thrown files=$(find "$od" -type f | wc -l)" >>"$LOG"
}

run bg_MAIN_a MAIN 300
run bg_MAIN_b MAIN 300

echo "=== $(date -u +%H:%M:%S) BG MAIN COMPLETE" >>"$LOG"

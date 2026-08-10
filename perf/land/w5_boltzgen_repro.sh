#!/bin/bash
# Exec pass 4. The unscored W5 cross-model run (perf/land/out/w5_crossmodel.log) shows something
# nobody has claimed: on arm L0 (main today) the BoltzGen design THROWS at device level
#
#   TT_THROW: Statically allocated circular buffers in program 51 clash with L1 buffers on core
#   range [(x=0,y=0) - (x=12,y=8)]. L1 buffer allocated at 1183744 and static circular buffer
#   region ends at 1405440
#
# after 19 s and exits rc=0 with 9 files, while arm L2 (main + W5's L1-residency guard fix) runs
# the same fixture for 40 minutes and writes 409. If that reproduces it is not a perf result, it is
# a crash on main that W5's guard fix happens to avoid, and it changes B's recommendation.
#
# Three runs, interleaved, fresh out dirs (the original driver SKIPs a dir that exists):
#   L0 -> expect the throw in ~20 s
#   L2 -> expect no throw; 7 minutes of clean running is conclusive against a 19 s failure
#   L0 -> repeat, because a one-off device state is the obvious alternative explanation
set -u
WT=/home/ttuser/.coworker/wt/perfwar-qb1-rebaseline-and-land
export TT_VISIBLE_DEVICES=3
export TT_BIO_LEASE_HOLDER=worker:perfwar-qb1-rebaseline-and-land
export TT_BIO_TRIMUL_OUT_FUSED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/usr/bin/python3
LOG="$WT/perf/land/out/w5_bg_repro.log"
OUTROOT=/home/ttuser/w5_xm

# Do not compete with the round-robin sweep for card 3.
while ! grep -q "SWEEP117RR COMPLETE" "$WT/perf/land/out/sweep117rr.log" 2>/dev/null; do sleep 15; done

run () {  # label arm cap
  local od="$OUTROOT/$1"
  rm -rf "$od"; mkdir -p "$od"
  echo "=== $(date -u +%H:%M:%S) $1 (arm $2, cap $3 s)" >>"$LOG"
  ( cd "$WT/arms/$2" && PYTHONPATH=$PWD timeout "$3" "$PY" -m tt_bio.main design examples/binder.yaml \
      --model boltzgen --seed 0 --out_dir "$od" >>"$LOG" 2>&1 )
  local rc=$?
  local thrown="no"
  grep -aq "TT_THROW" "$od/run.log" 2>/dev/null && thrown="YES"
  echo "=== $(date -u +%H:%M:%S) $1 rc=$rc tt_throw=$thrown files=$(find "$od" -type f | wc -l)" >>"$LOG"
}

run bg_L0_a L0 300
run bg_L2_a L2 420
run bg_L0_b L0 300

echo "=== $(date -u +%H:%M:%S) W5 BG REPRO COMPLETE" >>"$LOG"
grep -aE "^=== .* (rc=|COMPLETE)" "$LOG" | tail -8 >>"$LOG"

#!/bin/bash
# The quiet-regime control for the qb2 400-step ladder.
#
# Why it is needed. The first chain took the box at loadavg 0.04 and measured b=1 and b=8 there. At
# 00:38 another worker started a CPU parity reference on card 1, outside the benchlock, burning 7.5
# of the box's 16 cores, and b=2, b=4 and b=16 were all timed at loadavg 8-9 while it ran; benchlock
# logged its own "treat this run as suspect" warning when the second chain took over. That worker
# exited at about 02:15, before b=32 started, so the ladder splits cleanly into a loaded set
# {2, 4, 16} and a quiet set {1, 8, 32, 64}. Co-tenant noise on this box is 1-10 % and the b=8 vs
# b=16 gap the task exists to read is 4.4 %, so the two sets are not comparable as they stand.
#
# Re-running the loaded three on a quiet box, plus the two anchors as an A/A check that the quiet
# regime itself reproduces, puts the whole published curve in one regime. About forty minutes.
#
# Resumable: a rung whose JSON already carries warm_n is skipped, so relaunching after a worker pass
# ends does not repeat card time. Cheap rungs first.
set -u
WT=/home/ttuser/.coworker/wt/pxdesign-batchcurve-qb2-reverify
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OWNER=worker:pxdesign-batchcurve-qb2-reverify
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=$OWNER PYTHONPATH=$WT

done_already () { [ -f "$1" ] && $PY -c "import json,sys; sys.exit(0 if json.load(open('$1')).get('warm_n') else 1)" 2>/dev/null; }

run () {
  lab=$1; dir=$2; shift 2
  out="perf/newmodelcells/$dir/$lab.json"
  if done_already "$out"; then echo "=== $lab already measured, skipped"; return; fi
  echo "=== $lab start $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  $PY perf/newmodelcells/pxd_pagecell.py --tree "$WT" \
      --yaml perf/newmodelcells/laczc_512_tt.yaml "$@" --label "$lab" --out "$out"
  echo "=== $lab rc=$? $(date -u +%FT%TZ)"
}

run ctl400_n1  qb2_batchcurve400 --n_step 400 --n_sample 1  --rounds 4
run ctl400_n2  qb2_batchcurve400 --n_step 400 --n_sample 2  --rounds 4
run ctl400_n4  qb2_batchcurve400 --n_step 400 --n_sample 4  --rounds 4
run ctl400_n8  qb2_batchcurve400 --n_step 400 --n_sample 8  --rounds 4
run ctl400_n16 qb2_batchcurve400 --n_step 400 --n_sample 16 --rounds 3
echo "QB2_CTL400_DONE $(date -u +%FT%TZ)"

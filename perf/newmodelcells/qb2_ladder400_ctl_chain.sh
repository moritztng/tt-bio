#!/bin/bash
# The like-for-like control pair for the qb2 400-step ladder.
#
# Why it is needed. The first chain took the box at loadavg 0.04 and measured b=1 and b=8 on a quiet
# host. At 00:39 another worker started a CPU parity reference on card 1, outside the lock, burning
# 7.5 of 16 cores; every rung from b=16 onward was therefore timed at loadavg 8-9 and benchlock
# logged its own "treat this run as suspect" warning when the second chain took over. Co-tenant noise
# here is 1-10 % and the b=8 vs b=16 gap being read is 4.4 %, so the quiet rungs and the loaded rungs
# are not comparable as they stand.
#
# Re-running b=1 and b=8 at the end, in whatever regime then holds, makes the whole ladder one
# regime: either the pair reproduces the quiet numbers, in which case the co-tenant did not move
# anything and every rung stands, or it does not, and these two become the anchors the loaded rungs
# are read against. Twelve minutes of card time either way.
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

run ctl400_n1 qb2_batchcurve400 --n_step 400 --n_sample 1 --rounds 4
run ctl400_n8 qb2_batchcurve400 --n_step 400 --n_sample 8 --rounds 4
echo "QB2_CTL400_DONE $(date -u +%FT%TZ)"

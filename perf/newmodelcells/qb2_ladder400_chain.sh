#!/bin/bash
# The PXDesign batch ladder at 400 REAL steps on qb2 p300c card 0, resumable.
#
# Why this exists and the fit ladder does not answer it. On pc's p150a the ladder was fitted from
# n_step 8 and 24, and the fit reproduced a real 400-step run to 0.65 %. On qb2 the same two-point
# fit overshoots badly: from qb2's own s8/s24 rungs it predicts about 24.0 s/design at b=8 against
# 18.672 s actually measured at 400 steps. Short runs on this box are systematically slower per
# unit work, so the fit is not a licence here and every rung has to be run at full length.
#
# Resumable on purpose: a rung whose JSON already carries warm_n is skipped, so this can be
# relaunched after a worker pass ends without repeating an hour of card time. Cheap rungs first.
set -u
WT=/home/ttuser/.coworker/wt/pxdesign-batchcurve-qb2-reverify
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OWNER=worker:pxdesign-batchcurve-qb2-reverify
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=$OWNER PYTHONPATH=$WT

done_already () {  # done_already <json>  -> 0 if it already holds a finished run
  [ -f "$1" ] && $PY -c "import json,sys; sys.exit(0 if json.load(open('$1')).get('warm_n') else 1)" 2>/dev/null
}

run () {  # run <label> <outdir> <args...>
  lab=$1; dir=$2; shift 2
  out="perf/newmodelcells/$dir/$lab.json"
  if done_already "$out"; then echo "=== $lab already measured, skipped"; return; fi
  echo "=== $lab start $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  $PY perf/newmodelcells/pxd_pagecell.py --tree "$WT" \
      --yaml perf/newmodelcells/laczc_512_tt.yaml "$@" --label "$lab" --out "$out"
  echo "=== $lab rc=$? $(date -u +%FT%TZ)"
}

run c400_n2  qb2_batchcurve400 --n_step 400 --n_sample 2  --rounds 4
run c400_n4  qb2_batchcurve400 --n_step 400 --n_sample 4  --rounds 4
# The chunk ceiling against the batch ceiling, at full length so it compares to the 400-step rungs.
run mps8_c400_n32 qb2_batchchunk --n_step 400 --n_sample 32 --max_parallel_samples 8 --rounds 3
run c400_n32 qb2_batchcurve400 --n_step 400 --n_sample 32 --rounds 3
run c400_n64 qb2_batchcurve400 --n_step 400 --n_sample 64 --rounds 3
echo "QB2_LADDER400_DONE $(date -u +%FT%TZ)"

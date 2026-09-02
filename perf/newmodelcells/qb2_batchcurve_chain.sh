#!/bin/bash
# The PXDesign TT batch curve, re-measured on qb2 p300c card 0 under one benchlock hold.
#
# The pc-card0 run of 2026-08-26 built this curve on a p150a whose matmuls are known to be
# occasionally wrong (pc-card0-512aa-fold-nondeterminism). Timing is unaffected by that fault, so
# the SHAPE transferred, but no absolute second from that card may reach the page. This chain is
# the same harness, the same fixture and the same rungs on hardware the page is allowed to quote.
#
# Order is by value, not by section number: the two 400-step anchors first, then the ladder, then
# the b=16 rung the pc pass left owed, then the chunk-ceiling check. A chain cut short still leaves
# the amortisation factor measured.
set -u
WT=/home/ttuser/.coworker/wt/pxdesign-batchcurve-qb2-reverify
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OWNER=worker:pxdesign-batchcurve-qb2-reverify
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=$OWNER PYTHONPATH=$WT
CELL="$PY perf/newmodelcells/pxd_pagecell.py --tree $WT --yaml perf/newmodelcells/laczc_512_tt.yaml"

run () {  # run <label> <outdir> <args...>
  lab=$1; dir=$2; shift 2
  echo "=== $lab start $(date -u +%FT%TZ) load=$(cut -d" " -f1-3 /proc/loadavg)"
  $CELL "$@" --label "$lab" --out "perf/newmodelcells/$dir/$lab.json"
  echo "=== $lab rc=$? $(date -u +%FT%TZ)"
}

# 1. b=1 at 400 real steps: the absolute this box is allowed to publish, against the page cell.
run c400_n1 qb2_batchcurve400 --n_step 400 --n_sample 1 --rounds 4
# 2. b=8 at 400 real steps with rounds 5: seeds [0,1,2,3,0], so this single run is both the
#    amortisation anchor and the digest-reproduction check (round 4 repeats round 0s seed).
run d400_n8r5 qb2_batchcurve400 --n_step 400 --n_sample 8 --rounds 5
# 3. the ladder: two cheap step counts per rung, fitted to 400 by fit.py, cheap rungs first.
for S in 8 24; do
  for B in 1 2 4 8 16 32 64; do
    run "s${S}_n${B}" qb2_batchcurve --n_step "$S" --n_sample "$B" --rounds 3
  done
done
# 4. the rung pc left owed, at 400 real steps rather than fitted.
run c400_n16 qb2_batchcurve400 --n_step 400 --n_sample 16 --rounds 4
# 5. chunk ceiling against batch ceiling, in the fit regime the ladder was built in.
for S in 8 24; do
  run "mps8_s${S}_n32" qb2_batchchunk --n_step "$S" --n_sample 32 --max_parallel_samples 8 --rounds 3
done
echo "QB2_CHAIN_DONE $(date -u +%FT%TZ)"

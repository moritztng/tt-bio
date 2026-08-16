#!/bin/bash
# Blackhole neutrality A/B for the small-grid Transition row-chunk budget. On Blackhole the new
# branch is behind _IS_SMALL_GRID and can never be taken, so the arms are the two TREES rather than
# an env var: .bhbase is this branch's HEAD (unpatched), the worktree root is the patched arm.
# Legs interleaved; the two same-arm legs are the A/A floor.
WT=/home/ttuser/.coworker/wt/wh-perf-opendde
PY=/home/ttuser/tt-bio-dev/env/bin/python3.10
cd $WT || exit 1
mkdir -p $WT/perf/wh-opendde/results/bh_ab
leg() {  # label tree outfile
  TT_VISIBLE_DEVICES=1 TT_METAL_LOGGER_LEVEL=FATAL \
  TT_BIO_LEASE_HOLDER=worker:wh-perf-opendde \
    $PY perf/of3_4xpd/xmodel_ab.py --model opendde --tree $2 \
      --size 512 --repeat 3 --label $1 --out $3 > ${3%.json}.log 2>&1
  echo "EXIT $1 = $?" >> ${3%.json}.log
}
O=$WT/perf/wh-opendde/results/bh_ab
leg bh_patch_a $WT         $O/bh_patch_a.json
leg bh_base_a  $WT/.bhbase $O/bh_base_a.json
leg bh_patch_b $WT         $O/bh_patch_b.json
leg bh_base_b  $WT/.bhbase $O/bh_base_b.json
echo CHAIN_DONE > $O/.done

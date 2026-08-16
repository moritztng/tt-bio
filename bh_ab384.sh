#!/bin/bash
# Blackhole neutrality for BOTH levers at 384 aa, the size the SDPA band actually reaches.
# Arms are the two trees again: on 110 cores _IS_SMALL_GRID is False, so neither lever can change a
# code path and an env-var A/B here would be an A/A wearing an A/B label.
WT=/home/ttuser/.coworker/wt/wh-perf-opendde
PY=/home/ttuser/tt-bio-dev/env/bin/python3.10
cd $WT || exit 1
O=$WT/perf/wh-opendde/results/bh_ab384
mkdir -p $O
leg() {  # label tree outfile
  TT_VISIBLE_DEVICES=1 TT_METAL_LOGGER_LEVEL=FATAL \
  TT_BIO_LEASE_HOLDER=worker:wh-perf-opendde \
    $PY perf/of3_4xpd/xmodel_ab.py --model opendde --tree $2 \
      --size 384 --repeat 3 --label $1 --out $3 > ${3%.json}.log 2>&1
  echo "EXIT $1 = $?" >> ${3%.json}.log
}
leg bh384_patch_a $WT         $O/bh384_patch_a.json
leg bh384_base_a  $WT/.bhbase $O/bh384_base_a.json
leg bh384_patch_b $WT         $O/bh384_patch_b.json
leg bh384_base_b  $WT/.bhbase $O/bh384_base_b.json
echo CHAIN_DONE > $O/.done

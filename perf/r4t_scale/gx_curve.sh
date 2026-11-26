#!/bin/bash
# r4t-scale DP curve on anx-glx: real RFD4 step, widths 1..32, cards 0..w-1, 8 steps each, torch threads min(16, 64/w).
# Grades each width against the 1-chip leg with trunk_curve.py. Out ~/fleet/r4t/curve/w<w>/, curve_<w>.json.
R=~/fleet/r4t; O=$R/curve; mkdir -p $O; cd $R/r4t-scale
for w in 1 2 4 8 16 32; do
  [ -e ~/fleet/HANDBACK ] && { echo HANDBACK >> $O/chain.log; exit 4; }
  c=$(seq -s, 0 $((w-1))); t=$((64/w)); [ $t -gt 16 ] && t=16
  echo "w$w start $(date -u +%T) omp $t" >> $O/chain.log
  s=; [ $w -gt 1 ] && s=--shard; OMP=$t $R/gx_trunk.sh $c $O/w$w 8 $s; echo "w$w $(tail -1 $O/w$w/run.log)" >> $O/chain.log
  [ $w -gt 1 ] && ~/fleet/env/bin/python perf/r4t_scale/trunk_curve.py $O/w1 $O/w$w > $O/curve_$w.json 2>&1
done
echo DONE >> $O/chain.log

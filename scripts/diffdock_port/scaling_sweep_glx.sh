#!/bin/bash
# dfd-scaling on anx-glx (BH Galaxy, 64 cores): K=1/2/4/8 on chips 8-15, tile form then gemm, 3 reps, K alternating.
# Setup: tt-bio checkout at the form under test in ~/dfd-scaling/tt-bio, scaling_run.py beside it,
# the released checkpoint in ckpt/extracted and fx_8G43.pt in fx/ (both copied from qb2:~/scratch/dfd).
cd ~/dfd-scaling/tt-bio || exit 1
export DFD_CKPT=$HOME/dfd-scaling/ckpt/extracted PYTHONPATH=$PWD
OUT=$HOME/dfd-scaling/out
for form in tile gemm; do
  steps=190; [ $form = gemm ] && steps=57
  for r in 1 2 3; do for k in 1 2 4 8; do
    name=${form}_k${k}_r$r; [ -s $OUT/$name.json ] && continue
    [ -e ~/fleet/HANDBACK ] && { echo "HANDBACK, stop"; exit 0; }
    chips=$(seq -s, 8 $((7+k)))
    echo "-- $name chips $chips $(date -u +%FT%TZ) load: $(cut -d" " -f1-3 /proc/loadavg)"
    timeout 1500 ~/fleet/env/bin/python ../scaling_run.py --chips $chips --fixture ../fx/fx_8G43.pt \
      --steps $steps --poses 10 --edge-tp $form --threads 1 --rundir $OUT/rd_$name --out $OUT/$name.json > $OUT/$name.log 2>&1
    echo "-- $name rc=$? $(date -u +%FT%TZ) load: $(cut -d" " -f1-3 /proc/loadavg)"
  done; done
done
touch $OUT/DONE; echo "sweep done $(date -u +%FT%TZ)"

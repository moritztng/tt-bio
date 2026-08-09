#!/bin/sh
# Alternating A/B: the three model files at HEAD~1 (baseline) vs HEAD (batched_matmul).
# Alternating rather than blocked so host drift cannot land on one arm.
WT=/home/ttuser/.coworker/wt/perfwar-of3-matmul-sites
F="tt_bio/tenstorrent.py tt_bio/openfold3_atom_transformer.py tt_bio/openfold3_diffusion_transformer.py"
cd $WT || exit 1
for i in 1 2 3; do
  git checkout -q HEAD~1 -- $F
  sh perf/of3_mm/run_fold.sh ab_base_$i 0 > perf/of3_mm/logs/ab_base_$i.log 2>&1
  git checkout -q HEAD -- $F
  sh perf/of3_mm/run_fold.sh ab_fix_$i 0 > perf/of3_mm/logs/ab_fix_$i.log 2>&1
done
git checkout -q HEAD -- $F
echo AB_DONE

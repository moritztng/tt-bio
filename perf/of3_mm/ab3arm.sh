#!/bin/sh
# Three-arm alternating fold campaign: baseline vs in0_block_w=1 vs in0_block_w=2 (shipped).
#
# The two config variants were never measured against each other in one campaign; the earlier
# comparison was across campaigns hours apart, which on this host is worth several seconds of
# drift. Alternating all three inside one loop is the only way to attribute the difference to the
# constant rather than to the hour.
WT=/home/ttuser/.coworker/wt/perfwar-of3-matmul-sites
F="tt_bio/tenstorrent.py tt_bio/openfold3_atom_transformer.py tt_bio/openfold3_diffusion_transformer.py"
BASE=3a48eb33
cd $WT || exit 1
for i in 1 2 3; do
  git checkout -q $BASE -- $F
  sh perf/of3_mm/run_fold.sh k3_base_$i 0 > perf/of3_mm/logs/k3_base_$i.log 2>&1

  git checkout -q HEAD -- $F
  sed -i 's/^    return 2 if Kt > 2 and Kt % 2 == 0 else 1$/    return 1/' tt_bio/tenstorrent.py
  grep -q '^    return 1$' tt_bio/tenstorrent.py || { echo "K1_PATCH_FAILED"; exit 1; }
  sh perf/of3_mm/run_fold.sh k3_k1_$i 0 > perf/of3_mm/logs/k3_k1_$i.log 2>&1

  git checkout -q HEAD -- $F
  sh perf/of3_mm/run_fold.sh k3_k2_$i 0 > perf/of3_mm/logs/k3_k2_$i.log 2>&1
done
git checkout -q HEAD -- $F
echo K3_DONE

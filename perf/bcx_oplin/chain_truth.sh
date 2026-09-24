#!/bin/bash
# The 298-aa CDK2 monomer through fold_ab.py for each model, graded against PDB 1HCL.
cd /home/ttuser/.coworker/wt/bcx-oplin
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-oplin
for m in "$@"; do
  ~/tt-bio-dev/env/bin/python perf/bcx_oplin/fold_ab.py --model $m --target examples/prot300.yaml \
      --a3m scripts/gpu_vs_tt/fixtures/prot300.a3m --truth perf/bcx_oplin/1hcl.cif --seeds 3 \
      --out perf/bcx_oplin/folds/${m}_298 > perf/bcx_oplin/folds/${m}_298.log 2>&1
  echo "$m 298 rc=$? $(date -u +%FT%TZ)" >> perf/bcx_oplin/folds/chain.log
done

#!/bin/bash
# The ask-4649 control. cdk2x2_298 is monomeric CDK2 with no inter-domain hinge, so unlike
# cdk2x2_512 it can actually score a non-bit-exact change. Arms on,mm12,on in ONE process, CIFs
# kept by the harness. `mm12` here == the predecessor's `all`, because `ge` is merged into main.
WT=/home/ttuser/.coworker/wt/opendde-to-4x
cd $WT
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:opendde-to-4x PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_MAXLOAD=0.6
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh opendde-to-4x -- \
  $PY -u perf/other512/fold_ab_multi.py --model opendde --sizes 298 \
      --arms on,mm12,on --out perf/odde4x/ab_opendde_298_mm12.json
echo "RC=$?"

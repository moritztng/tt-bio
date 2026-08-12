#!/bin/bash
# The three runs section 9.5 of state/opendde-512aa-deep-perf.md specifies, in order, each under
# benchlock on qb2 card 2. A is the A/B that decides the pass; C is the cross-model gate on the
# divisor search and is cheap, so it goes before the long size sweep; B is the L1/DRAM capacity
# check at 576 (the last size taking g=12) and 640 (the fallback to 6).
WT=/home/ttuser/.coworker/wt/opendde-512aa-deep-perf
cd $WT || exit 1
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:opendde-512aa-deep-perf PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
export BENCHLOCK_MAXLOAD=0.6 BENCHLOCK_LOAD_WAIT_S=600
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh

echo "=== RUN A: opendde 512 aa, A/A pair + three single levers + the integrated arm ==="
$BL opendde-512aa-deep-perf -- \
  $PY -u perf/other512/fold_ab_multi.py --model opendde --sizes 512 \
      --arms on,on,g12,e6,mm12,all,on --out perf/odde512/ab_opendde_512.json
echo "RC_A=$?"

echo "=== RUN C: protenix-v2 512 aa, the divisor search must be byte-identical ==="
$BL opendde-512aa-deep-perf -- \
  $PY -u perf/other512/fold_ab_multi.py --model protenix-v2 --sizes 512 \
      --arms on,g12,on --out perf/odde512/ab_px_g12.json
echo "RC_C=$?"

echo "=== RUN B: opendde 576 + 640 aa, capacity and the group fallback ==="
$BL opendde-512aa-deep-perf -- \
  $PY -u perf/other512/fold_ab_multi.py --model opendde --sizes 576,640 \
      --arms on,all --out perf/odde512/ab_opendde_size.json
echo "RC_B=$?"
echo "=== ALL RUNS DONE ==="

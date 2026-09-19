#!/usr/bin/env bash
# Five seeds at 512 aa. Whole-molecule RMSD is not resolvable on cdk2x2_512 -- its hinge is
# unconstrained and the base-against-base seed floor runs into the tens of Angstrom -- so the
# question this answers is whether the stack's plDDT delta is inside the base's own across-seed
# plDDT scatter. One seed pair cannot say that.
set -u
WT=/home/ttuser/.coworker/wt/c13-land-first
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/c13_land
cd "$WT" || exit 1
exec /home/ttuser/.coworker/scripts/benchlock.sh c13-land-first -- env \
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:c13-land-first \
  "$PY" perf/c12_compose/acc_stack.py \
    --out "$OUT/acc512_seeds.json" --cifs "$OUT/acc512_seeds_cifs" \
    --size 512 --seeds 0,1,2,3,4 --arms base,both --mhz 1350

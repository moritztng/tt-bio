#!/usr/bin/env bash
# The 20 aa gate cell with its own paired A/A: off, off2, on per model, one process each,
# so the gate leg that guards the flip has a floor next to it instead of a bare number.
set -u
WT=/home/ttuser/.coworker/wt/shared-softmax-crossmodel-p3
PY=/home/ttuser/tt-bio-dev/env/bin/python
H=worker:shared-softmax-crossmodel-p3
L=$WT/perf/xmsoftmax/logs; mkdir -p "$L"
PTX=protenix.trunk,protenix.confidence
ODE=opendde.trunk,opendde.confidence,opendde.refiner

cell() {  # model sites arm card
  local sel=""
  [ "$3" = on ] && sel="$2"
  env TT_VISIBLE_DEVICES=$4 TT_BIO_LEASE_CARDS=$4 TT_BIO_LEASE_HOLDER=$H \
      TT_BIO_ACCURATE_SOFTMAX_AB="$sel" PYTHONPATH=$WT \
    $PY -u $WT/scripts/perf_regression.py --model $1 \
      > $L/cell20_$1_$3.log 2>&1
  echo "cell20 $1 $3 rc=$?" >> $L/driver.log
}
( for arm in off off2 on; do cell protenix-v2 "$PTX" $arm 2; done ) &
( for arm in off off2 on; do cell opendde "$ODE" $arm 3; done ) &
wait
echo "cell20 all done $(date -Is)" >> $L/driver.log

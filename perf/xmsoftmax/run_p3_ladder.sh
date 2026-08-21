#!/usr/bin/env bash
# Pass 3 ladder: the 768 aa rung for both models plus opendde 512 at two more reps.
# Two free cards on qb2 (2 and 3), one model each, so the pair fits one benchlock window.
set -u
WT=/home/ttuser/.coworker/wt/shared-softmax-crossmodel-p3
PY=/home/ttuser/tt-bio-dev/env/bin/python
H=worker:shared-softmax-crossmodel-p3
R=$WT/perf/xmsoftmax/results
L=$WT/perf/xmsoftmax/logs
mkdir -p "$R" "$L"

run() {  # card sites model sizes reps tag
  env TT_VISIBLE_DEVICES=$1 TT_BIO_LEASE_CARDS=$1 TT_BIO_LEASE_HOLDER=$H \
      TT_BIO_ACCURATE_SOFTMAX_AB=$2 PYTHONPATH=$WT \
    $PY -u $WT/perf/xmsoftmax/fold_ab_softmax.py --model $3 --sizes $4 --reps $5 \
      --instrument pf --out $R/fold_ab_$6.json > $L/$6.log 2>&1
  echo "done $6 rc=$?" >> $L/driver.log
}

PTX=protenix.trunk,protenix.confidence
ODE=opendde.trunk,opendde.confidence,opendde.refiner

echo "start $(date -Is) loadavg $(cat /proc/loadavg)" >> $L/driver.log
run 3 "$ODE" opendde 768 2 opendde_768 &
P3=$!
( run 1 "$PTX" protenix-v2 768 2 protenix_768; run 1 "$ODE" opendde 512 2 opendde_512_p3 ) &
P2=$!
wait $P3 $P2
echo "all done $(date -Is)" >> $L/driver.log

#!/bin/bash
# Runs ON the box: every model in turn, setup then arms, never two arms at once (session.sh pushed the weights vast cannot fetch).
#   bash box_chain.sh "openfold3 opendde protenix_v2 boltz2:off"   (model[:mode,mode]; default off,exact,fast)   markers CHAIN-PROGRESS, CHAIN-DONE, THERMAL-FAIL
cd /root/kc
for e in $1; do
  m=${e%%:*}; md=off,exact,fast; [ $m = $e ] || md=${e#*:}
  bash box_setup.sh $m > setup-$m.log 2>&1 || { echo $m setup-failed >> CHAIN-PROGRESS; continue; }
  [ -f burn.log ] || /root/kit/$m/venv/bin/python burn_gate.py > burn.log 2>&1 || { echo THERMAL-FAIL > THERMAL-FAIL; exit 3; }
  MODES=${md//,/ } bash box_arms.sh $m > arms-$m.log 2>&1
  echo $m $? >> CHAIN-PROGRESS
done
echo done > CHAIN-DONE

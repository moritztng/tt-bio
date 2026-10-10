#!/bin/bash
# Runs ON the box: every model in turn, setup then arms, never two arms at once (session.sh pushed the weights vast cannot fetch).
#   bash box_chain.sh "openfold3 opendde protenix_v2 boltz2:off"   (model[:mode,mode]; default off,exact,fast)   markers CHAIN-PROGRESS, CHAIN-DONE, THERMAL-FAIL
cd /root/kc
for e in $1; do
  m=${e%%:*}; md=off,exact,fast; [ $m = $e ] || md=${e#*:}
  bash box_setup.sh $m > setup-$m.log 2>&1 || { echo $m setup-failed >> CHAIN-PROGRESS; continue; }
  [ -f burn.log ] || /root/kit/$m/venv/bin/python burn_gate.py > burn.log 2>&1 || { echo THERMAL-FAIL > THERMAL-FAIL; exit 3; }
  # model:modes:levers_off, e.g. rosettafold3:exact,fast:prefetch,awrite
  lo=; case $md in *:*) lo=${md#*:}; md=${md%%:*};; esac
  MODES=${md//,/ } LEVERS_OFF=$lo bash box_arms.sh $m > arms-$m${lo:+-no}.log 2>&1
  echo $m $? >> CHAIN-PROGRESS
  # PRUNE=1: drop a measured kit's venvs (results stay) so the next kit's stack and weights fit the 150 GB disk
  [ -n "${PRUNE:-}" ] && rm -rf /root/kit/$m/venv /root/kit/$m/opt/venv SETUP-OK-$m
done
echo done > ${DONE_MARK:-CHAIN-DONE}

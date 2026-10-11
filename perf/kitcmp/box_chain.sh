#!/bin/bash
# Runs ON the box: every model in turn, setup then arms, never two arms at once (session.sh pushed the weights vast cannot fetch).
#   bash box_chain.sh "openfold3 opendde protenix_v2 boltz2:off"   (model[:mode,mode]; default off,exact,fast)   markers CHAIN-PROGRESS, CHAIN-DONE, THERMAL-FAIL
cd /root/kc
for e in $1; do
  m=${e%%:*}; md=off,exact,fast; [ $m = $e ] || md=${e#*:}
  SET=box_setup.sh ARMS=box_arms.sh BP=/root/kit/$m/venv/bin/python
  # the BindCraft section: two stacks of its own, and neither carries torch, so the burn gate gets its own venv
  [ $m = colabdesign ] && { SET=box_bc_setup.sh; ARMS=box_bc_arms.sh; BP=/opt/burn-venv/bin/python; }
  bash $SET $m > setup-$m.log 2>&1 || { echo $m setup-failed >> CHAIN-PROGRESS; continue; }
  [ -f burn.log ] || $BP burn_gate.py > burn.log 2>&1 || { echo THERMAL-FAIL > THERMAL-FAIL; exit 3; }
  # model:modes:levers_off, e.g. rosettafold3:exact,fast:prefetch,awrite
  lo=; case $md in *:*) lo=${md#*:}; md=${md%%:*};; esac
  MODES=${md//,/ } LEVERS_OFF=$lo bash $ARMS $m > arms-$m${lo:+-no}.log 2>&1
  echo $m $? >> CHAIN-PROGRESS
  # PRUNE=1: drop a measured kit's venvs (results stay) so the next kit's stack and weights fit the 150 GB disk
  [ -n "${PRUNE:-}" ] && rm -rf /root/kit/$m/venv /root/kit/$m/opt/venv SETUP-OK-$m
done
echo done > ${DONE_MARK:-CHAIN-DONE}

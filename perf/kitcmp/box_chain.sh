#!/bin/bash
# Runs ON the box: every model in turn, setup then arms, never two arms at once. protenix_v2 waits for CKPT-OK because
# its checkpoint URL answers 403 from vast hosts and the checkpoint is pushed from pc instead.
#   bash box_chain.sh "boltz2 openfold3 opendde protenix_v2"     markers CHAIN-PROGRESS, CHAIN-DONE, THERMAL-FAIL
cd /root/kc
for m in $1; do
  if [ $m = protenix_v2 ]; then while [ ! -f /root/kc/CKPT-OK ]; do sleep 20; done; fi
  bash box_setup.sh $m > setup-$m.log 2>&1 || { echo $m setup-failed >> CHAIN-PROGRESS; continue; }
  [ -f burn.log ] || /root/kit/$m/venv/bin/python burn_gate.py > burn.log 2>&1 || { echo THERMAL-FAIL > THERMAL-FAIL; exit 3; }
  bash box_arms.sh $m > arms-$m.log 2>&1
  echo $m $? >> CHAIN-PROGRESS
done
echo done > CHAIN-DONE

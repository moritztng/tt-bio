#!/bin/bash
# Runs ON the box: every model in turn, setup then arms, never two arms at once (session.sh pushed the weights vast cannot fetch).
#   bash box_chain.sh "openfold3 opendde protenix_v2 boltz2:off stock-saprot"   (model[:mode,mode]; default off,exact,fast)   markers CHAIN-PROGRESS, CHAIN-DONE, THERMAL-FAIL
cd /root/kc
# one GPU per session: on a multi-GPU box every arm, the burn gate and every nvidia-smi read take GPU 0 (PCI order)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
for e in $1; do
  m=${e%%:*}; md=off,exact,fast; [ $m = $e ] || md=${e#*:}
  # stock-<row>: a page row on its stock upstream stack (box_stock.sh), after the kits so the burn gate has run
  case $m in stock-*) bash box_stock.sh ${m#stock-} > $m.log 2>&1; echo $m $? >> CHAIN-PROGRESS; continue;; esac
  SET=box_setup.sh ARMS=box_arms.sh BP=/root/kit/$m/venv/bin/python
  # the BindCraft section: two stacks of its own, and neither carries torch, so the burn gate gets its own venv
  [ $m = colabdesign ] && { SET=box_bc_setup.sh; ARMS=box_bc_arms.sh; BP=/opt/burn-venv/bin/python; }
  bash $SET $m > setup-$m.log 2>&1 || { echo $m setup-failed >> CHAIN-PROGRESS; continue; }
  # a JAX kit (af2ig) has no torch either: same venv, created on first need
  $BP -c 'import torch' 2>/dev/null || { BP=/opt/burn-venv/bin/python; [ -x $BP ] || { for PB in /usr/local/bin/python3 /opt/py3.12.1/bin/python3 /opt/py3.12.10/bin/python3; do [ -x $PB ] && break; done
    $PB -m venv /opt/burn-venv &&
    $BP -m pip install -q torch --index-url https://download.pytorch.org/whl/cu130; } >> burn-venv.log 2>&1; }
  [ -f burn.log ] || $BP burn_gate.py > burn.log 2>&1 || { echo THERMAL-FAIL > THERMAL-FAIL; exit 3; }
  # model:modes:levers_off, e.g. rosettafold3:exact,fast:prefetch,awrite
  lo=; case $md in *:*) lo=${md#*:}; md=${md%%:*};; esac
  MODES=${md//,/ } LEVERS_OFF=$lo bash $ARMS $m > arms-$m${lo:+-no}.log 2>&1
  echo $m $? >> CHAIN-PROGRESS
  # PRUNE=1: drop a measured kit's venvs (results stay) so the next kit's stack and weights fit the 150 GB disk
  [ -n "${PRUNE:-}" ] && rm -rf /root/kit/$m/venv /root/kit/$m/opt/venv SETUP-OK-$m
done
echo done > ${DONE_MARK:-CHAIN-DONE}

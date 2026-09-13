#!/usr/bin/env bash
# The boltzgen designability leg, ON and OFF, interleaved, four runs.
#
# The 44-leg gate on this branch reproduced 43 of 44 leg details byte for byte against the
# gate_asg.json control and read boltzgen at 75 % / median 0.77972 against the control's
# 100 % / 0.80851. The leg designs FOUR binders, so its pass rate moves in 25 % steps and a
# single design landing above the 2.0 A scRMSD threshold moves it a whole step. This script asks
# whether the leg can tell the arms apart at all: the arms alternate ON/OFF/ON/OFF in one
# sequence so a box-load drift cannot land on one arm
# (memory op-ab-must-interleave-arms-compile-warmup-bias).
set -u
WT=/home/ttuser/.coworker/wt/b2z2-pwa-residency-ship
cd "$WT" || exit 1
export PYTHONPATH=$WT
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3
export ESM_ROOT=/home/ttuser/esm
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_CARDS=1
export TT_BIO_LEASE_HOLDER=worker:b2z2-pwa-residency-ship
for rep in 1 2; do
  for arm in on off; do
    if [ "$arm" = on ]; then v=1; else v=0; fi
    echo "=== rep $rep arm $arm (TT_BIO_PWA_BATCH_HEAD_WEIGHTS=$v) $(date -u +%H:%M:%SZ)"
    TT_BIO_PWA_BATCH_HEAD_WEIGHTS=$v /home/ttuser/tt-bio-dev/env/bin/python3 \
      scripts/full_parity_gate.py --leg boltzgen --fresh --workers localhost:1 \
      --load-ceiling 999 \
      --workdir "perf/b2z2_gate/bgab_${arm}${rep}_work" \
      --out "perf/b2z2_gate/gate_bgab_${arm}${rep}.json"
  done
done
echo "=== done $(date -u +%H:%M:%SZ)"

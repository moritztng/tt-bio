#!/bin/bash
# S1: the two levers that are live at 128/256 and were never measured there.
#   hchunk16  TRANSITION_H_CHUNK_SIZE_BIG 32 -> 16   (gated W<=384, so live at 128/256, dark at 512)
#   noL1out   _PAIR_PROJ_L1_OUT off
# Predicted before the run (state/boltz2-sizes-perf.md S1): hchunk16 SLOWER by 0.10-0.30 s at 128
# and 0.20-0.50 s at 256; noL1out within the A/A floor at both. If hchunk16 is FASTER than `on` by
# more than the A/A floor, the shipped 32 is a regression at this size class.
WT=/home/ttuser/.coworker/wt/boltz2-sizes-perf
cd $WT
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltz2-sizes-perf PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh boltz2-sizes-perf -- \
  $PY -u perf/other512/fold_ab_multi.py --model boltz2 --sizes 128,256 \
      --arms on,on,hchunk16,on,noL1out,on --out perf/b2sizes/s1_regress.json
RC=$?
echo "RC=$RC"

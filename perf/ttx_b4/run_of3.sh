#!/bin/bash
# Cross-model byte-identity check for the retune. openfold3 presents the (4,12), (4,16), (2,12)
# and (2,2) keys boltz2 512 never reaches, so its per-arm CIF digest is what proves the shared
# table stayed byte-for-byte for a second model.
WT=/home/ttuser/.coworker/wt/ttx-b4-genericop-kblock-ship
cd $WT
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:ttx-b4-genericop-kblock-ship
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
/home/ttuser/.coworker/scripts/benchlock.sh ttx-b4-genericop-kblock-ship -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/other512/fold_ab_multi.py --model openfold3 \
    --sizes 512 --arms on,mmretune,on,mmretune \
    --out perf/ttx_b4/ab_openfold3_512_qb2c3.json
echo "RC=$?"

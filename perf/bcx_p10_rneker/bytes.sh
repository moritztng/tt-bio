#!/bin/bash
# Leg 5: the two-arm byte census, untimed. Same block harness both arms, the flag the only
# difference, so a family that moves and is not the residual is the kernel doing something the
# flag did not advertise.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-p10-rneker
for arm in 0 1; do
    /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_devmap/devmap.py block \
        --n 288 --pad 288 --reps "${1:-1}" --rne-kernel $arm \
        --out ../bcx_p10_rneker/out/bytes_rne$arm.json
done

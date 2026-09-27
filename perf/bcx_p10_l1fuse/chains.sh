#!/bin/bash
# bcx-p10-l1fuse leg 1: the chain census, composed cell E, qb2 card 0.
set -euo pipefail
cd /home/ttuser/.coworker/wt/bcx-p10-l1fuse
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-p10-l1fuse
export TT_BIO_MM_LAYOUT=1
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_l1fuse/chains.py "$@"

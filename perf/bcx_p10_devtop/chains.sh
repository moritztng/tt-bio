#!/bin/bash
# bcx-p10-devtop leg 1: the l1fuse chain census on the tritraj stack (cell E, qb2 card 0).
# Same instrument as perf/bcx_p10_l1fuse/chains.sh; the stack levers stack5/tritraj arm are
# set here too, so the ranking is the round as it ships now. Extra env passes through (LEVER arm).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-0}
export TT_BIO_LEASE_HOLDER=worker:bcx-p10-devtop
export TT_BIO_MM_LAYOUT=1 TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_WIDEN_ADD=1
export TT_BIO_GRAD_FANIN_L1=0 TT_BIO_GENQ_COMPACT=0
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_l1fuse/chains.py "$@"

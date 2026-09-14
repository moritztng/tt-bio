#!/bin/sh
# whglx (j10glx02, Wormhole B0 Galaxy), card 2 -- this worker's grant.
cd ~/wt-difftx
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:roof-difftx-arith-efficiency
export TT_BIO_LEASE_DIR=/home/agent/leases TT_BIO_LEASE_TIMEOUT=900
export PYTHONPATH=/home/agent/wt-difftx
exec /home/agent/env/bin/python3 perf/roof_difftx/"$@" 2>&1 \
  | grep -vE "^ --- |DEBUG|info +\||^ *$|Degree|Total nodes|===|topology|Config\{"

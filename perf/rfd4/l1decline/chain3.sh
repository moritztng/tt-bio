#!/bin/bash
# After chain2: drain_parity at widths 2, 4, 8, 32 (the last is the refused plan, recorded as such).
R=/home/moritz/.cache/l1d-runs WT=/home/moritz/.coworker/wt/rfd4-l1decline
until [ -e $R/chain2.done ]; do sleep 15; done
cd /tmp/l1d-empty && PYTHONPATH=$WT TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:rfd4-l1decline \
  timeout 1500 ~/rfd4-fair/env/bin/python $WT/perf/rfd4/l1decline/drain_parity.py 2 4 8 32 > $R/drain_parity.log 2>&1
echo "rc=$? $(date -u +%FT%TZ)" >> $R/drain_parity.log

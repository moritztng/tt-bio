#!/bin/bash
# Queue openbind's record behind the rf3 1536 probe on chip 20, holding the chip from the start.
cd ~/wt-mgx-instrument
export TT_BIO_LEASE_DIR=$HOME/leases TT_BIO_LEASE_HOLDER=worker:mgx-instrument
~/env/bin/python perf/sizegate/mgx/hold.py 20 $$ >> perf/sizegate/mgx/logs/hold-20.log 2>&1 &
while pgrep -f "ladder_card.sh probe 20 rf3" >/dev/null; do sleep 20; done
exec perf/sizegate/mgx/ladder_card.sh record 20 openbind

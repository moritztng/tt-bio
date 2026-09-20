#!/bin/bash
# Card 0 is still finishing nesso1 from the first launch. Wait for that pid, then take a slice.
set -u
cd /home/ttuser/.coworker/wt/cov-ladder-below-bar-all-bhp150a || exit 1
while kill -0 3888875 2>/dev/null; do sleep 30; done
exec bash perf/sizegate/campaign/run_card.sh 0 esmfold2 boltz2 rf3 openfold3 protenix-v2 openbind opendde protenix-v1 nesso1

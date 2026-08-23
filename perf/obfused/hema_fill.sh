#!/bin/bash
# Wait for the prize sequence to finish (no device overlap), then fill the missing
# hema_512 def arm so step 4 pools four targets instead of three.
while pgrep -f "perf/obfused/prize.sh" >/dev/null; do sleep 20; done
cd /home/ttuser/.coworker/wt/openbind-fused-sdpa-rescore
bash perf/obfused/disto512_of3.sh 1 0,1,2 hema_512

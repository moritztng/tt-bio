#!/bin/bash
cd /home/ttuser/.coworker/wt/of3t-fullstep64
bash perf/of3t_fullstep64/devarm.sh ON1 on weights
bash perf/of3t_fullstep64/devarm.sh OFF off
bash perf/of3t_fullstep64/devarm.sh ON2 on
echo CHAIN_DONE

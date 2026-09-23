#!/usr/bin/env bash
# of3t-inproj qb2 card 1: ON with the walked weights (for the bijection), its A/A, and PADON
# (+3 on the pad-token input features), which must leave every gradient where ON put it.
cd /home/ttuser/.coworker/wt/of3t-inproj
S=/home/ttuser/of3t_inproj
bash perf/of3t_inproj/devarm.sh INON on weights > $S/devarm_INON.log 2>&1
bash perf/of3t_inproj/devarm.sh INON2 on > $S/devarm_INON2.log 2>&1
DEVSTEP_EXTRA="--pad-perturb 3" bash perf/of3t_inproj/devarm.sh PADON on > $S/devarm_PADON.log 2>&1
echo "=== chain done $(date -u +%FT%TZ)" >> $S/chain_dev.log

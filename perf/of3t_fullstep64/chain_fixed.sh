#!/usr/bin/env bash
# of3t-fullstep64: the mask fix in tt_bio/train/openfold3.py. FIXOFF must reproduce the diagnostic
# arm M384OFF bit-identically; PADOFF adds 3 to the pad-token input features and must not move a gradient.
cd /home/ttuser/.coworker/wt/of3t-fullstep64
bash perf/of3t_fullstep64/devarm.sh FIXOFF off > /home/ttuser/of3t_fullstep64/devarm_FIXOFF.log 2>&1
DEVSTEP_EXTRA="--pad-perturb 3" bash perf/of3t_fullstep64/devarm.sh PADOFF off > /home/ttuser/of3t_fullstep64/devarm_PADOFF.log 2>&1
echo "=== chain done $(date -u +%FT%TZ)" >> /home/ttuser/of3t_fullstep64/chain_fixed.log

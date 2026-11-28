#!/usr/bin/env bash
# The row's two remaining bars, serialised: pc is oversubscribed (load 63 on 12 cores), so
# running these beside the grade arms makes everything slower, not faster.
#   1. wait out the production grade arms (7BJ6_TVK control, then 7F8T_FAD via chain_fad.sh)
#   2. the equivariance re-run with the guard in place, per axis
set -u
cd /home/moritz/.coworker/wt/dfd-knn32
while pgrep -f "score_net_parity[.]py|score_net_grade[.]py" > /dev/null; do sleep 60; done
echo "=== grade arms done $(date -Is), equivariance re-run starts"
TT_VISIBLE_DEVICES= PYTHONPATH=. nice -n 5 /home/moritz/tt-bio/env/bin/python3.10 \
  -m pytest tests/test_diffdock_equivariance.py -q -s
echo "=== equivariance rc=$? $(date -Is)"

#!/usr/bin/env bash
# The four gradient arms, one per idle card. Card 1 is the grant; 0/2/3 were verified idle
# (no process holding /dev/tenstorrent/*) before this ran, and the lease is widened per command.
set -uo pipefail
cd /home/ttuser/.coworker/wt/of3t-auxgrad
L=/home/ttuser/of3t_auxgrad_logs; mkdir -p $L
CARD=1 bash perf/of3t_auxgrad/gradrun.sh grad_M_masked   --mask --shipped-forward \
    --dump-grads /home/ttuser/of3t_auxgrad_logs/grads_M.pt   > $L/M.log 2>&1 &
p1=$!
CARD=2 bash perf/of3t_auxgrad/gradrun.sh grad_N_unmasked --shipped-forward \
    --dump-grads /home/ttuser/of3t_auxgrad_logs/grads_N.pt   > $L/N.log 2>&1 &
p2=$!
CARD=3 bash perf/of3t_auxgrad/gradrun.sh grad_M_zero     --mask --zero-model  > $L/Z.log 2>&1 &
p3=$!
CARD=0 bash perf/of3t_auxgrad/gradrun.sh grad_M_scramble --mask --scramble-cot > $L/S.log 2>&1 &
p4=$!
wait $p1; echo "M exit $?"
wait $p2; echo "N exit $?"
wait $p3; echo "Z exit $?"
wait $p4; echo "S exit $?"
echo ARMS_ALLDONE $(date -u +%FT%TZ)

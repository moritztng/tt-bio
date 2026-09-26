#!/usr/bin/env bash
# of3t-p10fp32bw: wait out whatever diffusion arm is on card 1 right now, then run the chain.
# Detached deliberately: the pair of scope arms is ~36 min of card plus two CPU rescores, which
# does not fit one worker turn, and an arm restarted from zero every pass never finishes.
# cwd is this row's OWN tree, per the fleet rule about jobs rooted in a worktree that may be
# torn down under them.
set -uo pipefail
O=/home/ttuser/of3t_p10fp32bw
cd "$O/tree"
exec >> "$O/supervise.log" 2>&1
echo "=== supervisor start $(date -u +%FT%TZ) pid $$ ==="
for i in $(seq 1 240); do
  # The supervisor's own argv does not contain the pattern, so this cannot match itself.
  if ! pgrep -f 'perf/of3t_diffusion/device_gradient.py' > /dev/null; then
    echo "no device_gradient.py alive at $(date -u +%FT%TZ) after ${i} checks"
    break
  fi
  sleep 15
done
sleep 5
exec bash "$O/chain.sh"

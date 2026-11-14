#!/bin/bash
# The issue #21 card RERUN, queued so it does not stall the row ahead of it.
#
# state/bci/CHIPS.md puts bci-memory's four short legs first and this rerun next, and
# bci-memory's chain.sh calls quiet() BEFORE each leg, waiting on any live
# bci_validation/card_campaign.py. So launching now would hold chain.sh's legs 2-4 for hours.
# Instead this waits until chain.sh has TAKEN the lock for its last short leg (test_before),
# then queues on the same flock. chain.sh's long leg then sees this process in its quiet()
# and holds, which is the order CHIPS.md asks for.
#
#   setsid nohup perf/bci_validation/rerun_when_memory_reaches_test_before.sh 1 \
#       >/home/ttuser/bci_card_fixed2.log 2>&1 &
set -uo pipefail
card=${1:?the card the grant in state/bci/CHIPS.md names}
traj=${BCI_TRAJ:-8}
memout=/home/ttuser/.coworker/wt/bci-memory/perf/bc2_memory/out
here=$(cd "$(dirname "$0")" && pwd)

# At most 5 h of waiting: if chain.sh never reaches test_before, take the chip anyway.
deadline=$(( $(date +%s) + 18000 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  # go as soon as the last short leg has the lock, or if the chain is already past it
  compgen -G "$memout/test_before.locked.*" >/dev/null && break
  compgen -G "$memout/long.locked.*" >/dev/null && break
  [ -e "$memout/test_before.rc" ] && break
  pgrep -f 'perf/bc2_memory/chain.sh' >/dev/null || break   # chain gone, chip is ours
  sleep 20
done
echo "=== queueing the rerun $(date -u +%FT%TZ), ${traj} trajectories"
exec env BCI_HOURS=5 "$here/card_run.sh" fixed "$card" --max-trajectories "$traj"

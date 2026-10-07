#!/bin/bash
# Queue run.sh behind the bci-validation rerun, the order state/bci/CHIPS.md sets for chip 1.
# Validation's python only exists once it holds the chip lock, so queueing on the lock after
# seeing it puts this run next: bci-memory's chain.sh polls quiet() every 60 s and queues its
# long leg only after validation has exited.
#   setsid nohup perf/bci_refusal/after_validation.sh r1 > /home/ttuser/bci_refusal_wait.log 2>&1 < /dev/null &
set -uo pipefail
tag=${1:?tag}
here=$(cd "$(dirname "$0")" && pwd)
memout=/home/ttuser/.coworker/wt/bci-memory/perf/bc2_memory/out
validation_running() {
  local p
  for p in $(pgrep -f bci_validation/card_campaign.py); do
    [ "$(cat /proc/$p/comm 2>/dev/null)" = python3 ] && return 0
  done
  return 1
}
deadline=$(( $(date +%s) + 21600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  validation_running && break
  # validation never queued: go once bci-memory's short legs are done
  if ! pgrep -f rerun_when_memory_reaches_test_before >/dev/null && [ -e "$memout/test_before.rc" ]; then break; fi
  sleep 20
done
echo "=== queueing $tag $(date -u +%FT%TZ)"
exec "$here/run.sh" "$tag" wait

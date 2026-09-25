#!/usr/bin/env bash
# Pull the running continuation back every 60 s, then destroy the box. Teardown is in an EXIT trap
# so a crash in the loop still ends the billing.
set -uo pipefail
INSTANCE=52585411
HOST=ssh6.vast.ai
PORT=25410
KEY=/home/moritz/.ssh/id_ed25519
RESULTS=/home/moritz/.coworker/wt/bcx-gpuref/perf/bcx_gpuref/results/warm
LEDGER=$RESULTS/rental.txt
VAST=/home/moritz/.vast-venv/bin/vastai
VAST_API_KEY=$(cat /home/moritz/.config/vastai/vast_api_key); export VAST_API_KEY
BUDGET_FILE=/home/moritz/.coworker/state/vast-budget
DEADLINE=$(( $(date +%s) + 2100 ))   # 35 min hard stop, inside this turn and inside the 2.0 h cap

log() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$LEDGER"; }
SSH=(ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 -i "$KEY" -p "$PORT" "root@$HOST")
pull() {
  rsync -az -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i $KEY -p $PORT" "root@$HOST:/root/bcx_out/" "$RESULTS/" 2>/dev/null
  rsync -az -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i $KEY -p $PORT" "root@$HOST:/root/run_warm2.log" "$RESULTS/" 2>/dev/null
  rsync -az -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i $KEY -p $PORT" "root@$HOST:/root/install2.log" "$RESULTS/" 2>/dev/null
}
teardown() {
  log "final pull before teardown"
  pull
  "${SSH[@]}" 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv' > "$RESULTS/box_after.txt" 2>&1
  for a in 1 2 3 4 5; do
    "$VAST" destroy instance "$INSTANCE" -y 2>&1 | tee -a "$LEDGER" | grep -qi 'destroying\|success' && break
    sleep 10
  done
  sleep 8
  if "$VAST" show instances --raw 2>/dev/null | grep -q "\"id\": $INSTANCE,"; then
    log "STILL PRESENT after destroy: $INSTANCE -- destroy by hand"
  else
    log "teardown confirmed: instance $INSTANCE is gone"
  fi
  sed -i "/^$INSTANCE[[:space:]]/d" "$BUDGET_FILE" 2>/dev/null
}
trap teardown EXIT

log "watching continuation on $INSTANCE (14 trajectories); hard stop in 35 min"
while true; do
  pull
  if ! "${SSH[@]}" 'pgrep -f bcx_gpuref/run_on_box > /dev/null' 2>/dev/null; then
    log "the continuation has finished"; break
  fi
  if [[ $(date +%s) -ge $DEADLINE ]]; then
    log "35-min watch ceiling reached; stopping with what is measured"; break
  fi
  sleep 60
done

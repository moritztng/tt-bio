#!/usr/bin/env bash
# Start the vast.ai box this row already owns, measure BindCraft 2 on it, pull the results back and
# STOP it. One command, and the stop is in an EXIT trap so a crash cannot leak the credit.
#
#   bash start_and_measure.sh                    # the matched arm: 4 trajectories at binder 146
#   ARM=stock BINDER_LENGTH= MAX_TRAJECTORIES=3 bash start_and_measure.sh   # stock length mix
#   DRY_RUN=1 bash start_and_measure.sh          # ledger and ssh probe only, starts nothing
#
# It STOPS rather than destroys. The 500 GB disk costs $0.45/month and carries this row's BC2
# install and the 5.3 GB AlphaFold parameters, so a stop is a pause and a destroy is another
# install. `vastai start instance <id>` resumes.
#
# Three things it gets right that improvising on a billing box would not:
#
#   1. The stop is unconditional and verified. It retries five times and then reads the account
#      back, because a stop that lost a race bills all night at $11.45/hr.
#   2. Results rsync back every RSYNC_EVERY seconds while the run is going, so hitting the ceiling
#      costs the remaining trajectories and not the measured ones.
#   3. The dollar ceiling is a wall-clock deadline computed from the instance's own rate and
#      enforced here, below the 3.0 h vast_guard.sh would stop it at.
set -euo pipefail

INSTANCE=${INSTANCE:-52151627}
BUDGET_H=${BUDGET_H:-2.2}
RSYNC_EVERY=${RSYNC_EVERY:-90}
ARM=${ARM:-warm}
LABEL=${LABEL:-bcx-gpuref}
RESTORE_LABEL=${RESTORE_LABEL:-mgx-reference}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESULTS=${RESULTS:-$HERE/results/$ARM}
VAST=${VAST:-/home/moritz/.vast-venv/bin/vastai}
KEY_FILE=${KEY_FILE:-/home/moritz/.config/vastai/vast_api_key}
SSH_KEY=${SSH_KEY:-/home/moritz/.ssh/id_ed25519}
GUARD_STATE=${GUARD_STATE:-/home/moritz/.coworker/state/vast_guard}

mkdir -p "$RESULTS"
LEDGER="$RESULTS/rental.txt"
VAST_API_KEY=$(cat "$KEY_FILE"); export VAST_API_KEY

log() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$LEDGER"; }

STARTED_AT=""
RATE=0
stop_box() {
  local status=$?
  if [[ -n $STARTED_AT ]]; then
    # This box is shared: mgx-reference's worker is live and its 26 remaining reference folds run
    # here too. Stopping it under a fold in flight would destroy hours of another row's work, so
    # the stop is skipped while a foreign compute process holds a card -- loudly, and with
    # vast_guard.sh's 3.0 h current-run rail left as the backstop rather than nothing.
    BUSY=""
    if [[ -n ${HOST:-} && -n ${PORT:-} ]]; then
      BUSY=$(ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -i "$SSH_KEY" -p "$PORT" \
             "root@$HOST" 'nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader' 2>/dev/null \
             | grep -v 'run_on_box\|bc2_step_timing' || true)
    fi
    if [[ -n ${BUSY//[[:space:]]/} ]]; then
      log "NOT stopping instance $INSTANCE: a foreign compute process still holds a card -- $BUSY"
      bash /home/moritz/.coworker/tg.sh send "bcx-gpuref: measurement done, but instance 52151627 has a foreign GPU process ($BUSY) so I did NOT stop it. Label left as $LABEL. vast_guard's 3.0 h rail is the only thing bounding it now." >/dev/null 2>&1 || true
      log "exit status $status; results in $RESULTS"
      return $status
    fi
    log "stopping instance $INSTANCE"
    for attempt in 1 2 3 4 5; do
      "$VAST" stop instance "$INSTANCE" >> "$LEDGER" 2>&1 && break
      sleep 10
    done
    sleep 8
    state=$("$VAST" show instance "$INSTANCE" --raw 2>/dev/null | python3 -c 'import json,sys
d = json.load(sys.stdin)
print(d.get("actual_status") or d.get("cur_state") or "?")' 2>/dev/null || echo "?")
    if [[ $state == running ]]; then
      log "STILL RUNNING after stop: instance $INSTANCE at \$$RATE/hr -- stop it by hand"
    else
      log "stop confirmed: instance $INSTANCE reads '$state'"
    fi
    "$VAST" label instance "$INSTANCE" "$RESTORE_LABEL" >> "$LEDGER" 2>&1 || true
    local hours
    hours=$(python3 -c "import time; print(round((time.time() - $STARTED_AT) / 3600, 3))")
    log "billed window ${hours}h at \$$RATE/hr = about \$$(python3 -c "print(round($RATE * $hours, 2))")"
  fi
  log "exit status $status; results in $RESULTS"
  return $status
}
trap stop_box EXIT

read -r RATE STATE0 GPUS < <("$VAST" show instance "$INSTANCE" --raw 2>/dev/null | python3 -c 'import json,sys
d = json.load(sys.stdin)
print(round(d.get("dph_total") or 0, 4), d.get("actual_status") or "?", d.get("num_gpus") or "?")')
CREDIT=$("$VAST" show user --raw 2>/dev/null | python3 -c 'import json,sys; print(round(json.load(sys.stdin).get("credit",0),2))')
log "instance $INSTANCE: $GPUS GPU(s) at \$$RATE/hr, reads '$STATE0'; credit \$$CREDIT; ceiling ${BUDGET_H}h = about \$$(python3 -c "print(round($RATE * $BUDGET_H, 2))")"

if [[ -n ${DRY_RUN:-} ]]; then log "DRY_RUN: nothing started"; exit 0; fi

# vast_guard.sh reads the CURRENT-RUN age out of state/vast_guard/<id>.spell, and falls back to the
# instance's CREATION date when it has never seen the instance running. This box was created 58 h
# ago, so that fallback would read 58 h against a 3.0 h budget and stop the run inside one tick.
# Writing the spell now is the truthful value: this is when the current run begins.
mkdir -p "$GUARD_STATE"
python3 -c "import time,sys; open('$GUARD_STATE/$INSTANCE.spell','w').write(repr(time.time())); open('$GUARD_STATE/$INSTANCE.seen','w').write(str(int(time.time())))"
"$VAST" label instance "$INSTANCE" "$LABEL" >> "$LEDGER" 2>&1 || true

"$VAST" start instance "$INSTANCE" 2>&1 | tee -a "$LEDGER"
STARTED_AT=$(date +%s)
DEADLINE=$(python3 -c "print(int($STARTED_AT + $BUDGET_H * 3600))")

# Wait for ssh rather than for a status field: the field goes green before sshd answers.
HOST=""; PORT=""
for attempt in $(seq 1 60); do
  read -r HOST PORT < <("$VAST" show instance "$INSTANCE" --raw 2>/dev/null | python3 -c 'import json,sys
d=json.load(sys.stdin); print(d.get("ssh_host") or "", d.get("ssh_port") or "")' || echo " ")
  if [[ -n ${HOST:-} && -n ${PORT:-} ]] && ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
      -i "$SSH_KEY" -p "$PORT" "root@$HOST" true 2>/dev/null; then
    log "ssh up at $HOST:$PORT after $((attempt * 10))s"
    break
  fi
  [[ $attempt -eq 60 ]] && { log "ssh never came up in 10 min"; exit 1; }
  sleep 10
done

SSH=(ssh -o StrictHostKeyChecking=no -o BatchMode=yes -i "$SSH_KEY" -p "$PORT" "root@$HOST")
"${SSH[@]}" 'mkdir -p /root/bcx_gpuref'
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" -P "$PORT" -q \
  "$HERE"/{bc2_step_timing.py,analyze_steps.py,validate_designs.py,test_bc2_contract.py,run_on_box.sh} \
  "root@$HOST:/root/bcx_gpuref/"
log "harness copied"

# Who else is on this box. GPU 1 belongs to nobody in this run, and a foreign process on GPU 0
# would make every number here an artifact, so the answer is recorded before and after.
"${SSH[@]}" 'nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv; nvidia-smi --query-gpu=index,name,clocks.sm,utilization.gpu --format=csv' \
  > "$RESULTS/box_before.txt" 2>&1 || true
cat "$RESULTS/box_before.txt"

"${SSH[@]}" "ARM=$ARM BINDER_LENGTH=${BINDER_LENGTH-146} MAX_TRAJECTORIES=${MAX_TRAJECTORIES:-4} FINAL_DESIGNS=${FINAL_DESIGNS:-4} setsid nohup bash /root/bcx_gpuref/run_on_box.sh > /root/run_$ARM.log 2>&1 < /dev/null &" || true
log "run started on the box"

while true; do
  rsync -az -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $PORT" "root@$HOST:/root/bcx_out/" "$RESULTS/" 2>/dev/null || true
  rsync -az -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $PORT" "root@$HOST:/root/run_$ARM.log" "$RESULTS/" 2>/dev/null || true
  if ! "${SSH[@]}" 'pgrep -f run_on_box.sh > /dev/null' 2>/dev/null; then
    log "the run has finished"
    break
  fi
  if [[ $(date +%s) -ge $DEADLINE ]]; then
    log "${BUDGET_H}h ceiling reached; stopping with what is measured"
    break
  fi
  sleep "$RSYNC_EVERY"
done

"${SSH[@]}" 'nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv' > "$RESULTS/box_after.txt" 2>&1 || true
rsync -az -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $PORT" "root@$HOST:/root/bcx_out/" "$RESULTS/" 2>/dev/null || true
rsync -az -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $PORT" "root@$HOST:/root/run_$ARM.log" "$RESULTS/" 2>/dev/null || true
log "results pulled to $RESULTS"

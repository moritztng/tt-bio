#!/usr/bin/env bash
# Rent an H200, measure BindCraft 2's gradient step on it, pull the results back, destroy it.
#
#   bash rent_and_measure.sh                 # rent the cheapest H200 and run
#   OFFER=50416319 bash rent_and_measure.sh  # a named offer
#   DRY_RUN=1 bash rent_and_measure.sh       # print what it would rent, create nothing
#
# Three things this exists to get right, none of which survive being improvised on a billing box:
#
#   1. The instance id is written to disk the moment it exists, before anything can fail, and the
#      teardown runs from an EXIT trap. A forgotten instance bills until someone notices; this
#      account is already $4 negative because one sat stopped.
#   2. Results are pulled back every RSYNC_EVERY seconds while the run is going, so a stop at the
#      budget ceiling loses nothing already measured. mgx-reference learned this the expensive way.
#   3. A wall-clock ceiling derived from the dollar ceiling, enforced here rather than trusted to
#      the campaign ending on its own.
set -euo pipefail

BUDGET_USD=${BUDGET_USD:-60}
RSYNC_EVERY=${RSYNC_EVERY:-120}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS=${RESULTS:-$HERE/results/$STAMP}
VAST=${VAST:-/home/moritz/.vast-venv/bin/vastai}
KEY_FILE=${KEY_FILE:-/home/moritz/.config/vastai/vast_api_key}
IMAGE=${IMAGE:-pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime}
SSH_KEY=${SSH_KEY:-/home/moritz/.ssh/id_ed25519}

mkdir -p "$RESULTS"
LEDGER="$RESULTS/rental.txt"
VAST_API_KEY=$(cat "$KEY_FILE"); export VAST_API_KEY

log() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$LEDGER"; }

INSTANCE=""
teardown() {
  local status=$?
  if [[ -n $INSTANCE ]]; then
    log "destroying instance $INSTANCE"
    # Tried until it takes: an instance left up because one API call lost a race bills all night.
    for attempt in 1 2 3 4 5; do
      if "$VAST" destroy instance "$INSTANCE" 2>&1 | tee -a "$LEDGER" | grep -qi 'destroy\|success'; then break; fi
      sleep 10
    done
    sleep 5
    if "$VAST" show instances --raw 2>/dev/null | grep -q "\"id\": $INSTANCE,"; then
      log "STILL PRESENT after destroy: instance $INSTANCE -- destroy it by hand"
    else
      log "teardown confirmed: instance $INSTANCE is gone"
    fi
  fi
  log "exit status $status; results in $RESULTS"
  return $status
}
trap teardown EXIT

# The offer, and what it costs. The hourly rate goes in the ledger because a figure whose cost is
# not on the record cannot be weighed against the next one.
if [[ -z ${OFFER:-} ]]; then
  read -r OFFER RATE < <("$VAST" search offers 'gpu_name=H200 num_gpus=1 rentable=true disk_space>=150 cuda_max_good>=12.4 inet_down>=200' -o 'dph+' --raw 2>/dev/null \
    | python3 -c 'import json,sys; o=json.load(sys.stdin); print(o[0]["id"], round(o[0]["dph_total"],4)) if o else sys.exit("no H200 offer")')
else
  RATE=$("$VAST" search offers "id=$OFFER" --raw 2>/dev/null | python3 -c 'import json,sys; o=json.load(sys.stdin); print(round(o[0]["dph_total"],4))')
fi
HOURS=$(python3 -c "print(round($BUDGET_USD / max($RATE, 0.0001), 2))")
DEADLINE=$(python3 -c "import time; print(int(time.time() + $HOURS * 3600))")
log "offer $OFFER at \$$RATE/hr; \$$BUDGET_USD ceiling is ${HOURS}h"

if [[ -n ${DRY_RUN:-} ]]; then
  log "DRY_RUN: nothing created"
  INSTANCE=""
  exit 0
fi

CREATED=$("$VAST" create instance "$OFFER" --image "$IMAGE" --disk 150 --ssh --direct --label bcx-gpuref --raw 2>&1 | tee -a "$LEDGER")
INSTANCE=$(echo "$CREATED" | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("new_contract","") or "")
except Exception: print("")')

# If the response parsed to nothing the instance may still EXIST, and an instance we cannot name is
# one we cannot destroy -- the precise way this account ended up $4 negative. So ask the account
# what it is running under our label before concluding there is nothing to tear down.
if [[ -z $INSTANCE ]]; then
  log "no id in the create response; asking the account what is running under label bcx-gpuref"
  sleep 10
  INSTANCE=$("$VAST" show instances --raw 2>/dev/null | python3 -c 'import json,sys
found=[i["id"] for i in json.load(sys.stdin) if i.get("label") == "bcx-gpuref"]
print(found[-1] if found else "")')
fi
[[ -n $INSTANCE ]] || { log "create returned no id and nothing is running under the label; nothing was rented"; exit 1; }
echo "$INSTANCE" > "$RESULTS/instance_id.txt"
log "instance $INSTANCE created"

# Wait for ssh rather than for a status field: the field goes green before sshd answers.
for attempt in $(seq 1 60); do
  read -r HOST PORT < <("$VAST" show instance "$INSTANCE" --raw 2>/dev/null | python3 -c 'import json,sys
d=json.load(sys.stdin); print(d.get("ssh_host",""), d.get("ssh_port",""))' || echo " ")
  if [[ -n ${HOST:-} && -n ${PORT:-} ]] && ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
      -i "$SSH_KEY" -p "$PORT" "root@$HOST" true 2>/dev/null; then
    log "ssh up at $HOST:$PORT after $((attempt * 15))s"
    break
  fi
  [[ $attempt -eq 60 ]] && { log "ssh never came up"; exit 1; }
  sleep 15
done

SSH=(ssh -o StrictHostKeyChecking=no -o BatchMode=yes -i "$SSH_KEY" -p "$PORT" "root@$HOST")
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" -P "$PORT" -q \
  "$HERE"/{bc2_step_timing.py,analyze_steps.py,validate_designs.py,test_bc2_contract.py,run_on_box.sh} \
  "root@$HOST:/root/"
log "harness copied"

"${SSH[@]}" "BINDER_LENGTH=${BINDER_LENGTH:-96} HERE=/root nohup bash /root/run_on_box.sh > /root/run.log 2>&1 &" || true

# Pull results back while it runs, so a stop at the ceiling loses nothing already measured.
while true; do
  rsync -az -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $PORT" "root@$HOST:/root/bcx_out/" "$RESULTS/" 2>/dev/null || true
  rsync -az -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $PORT" "root@$HOST:/root/run.log" "$RESULTS/" 2>/dev/null || true
  if ! "${SSH[@]}" 'pgrep -f run_on_box.sh > /dev/null' 2>/dev/null; then
    log "the run has finished"
    break
  fi
  if [[ $(date +%s) -ge $DEADLINE ]]; then
    log "\$$BUDGET_USD ceiling reached after ${HOURS}h; stopping with what is measured"
    break
  fi
  sleep "$RSYNC_EVERY"
done

rsync -az -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $PORT" "root@$HOST:/root/bcx_out/" "$RESULTS/" 2>/dev/null || true
ELAPSED_H=$(python3 -c "import time,os; print(round((time.time() - os.path.getmtime('$RESULTS/instance_id.txt')) / 3600, 3))")
log "instance $INSTANCE, \$$RATE/hr, ${ELAPSED_H}h, about \$$(python3 -c "print(round($RATE * $ELAPSED_H, 2))")"

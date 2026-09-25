#!/usr/bin/env bash
# The 1536 seed floor on the real complex, on a rented box, start to stop in one script.
#
#   bash perf/mgx/ref/ref_1536_floor.sh            # run it
#   MGX_MAX_MIN=90 bash perf/mgx/ref/ref_1536_floor.sh
#
# Why only 3abq_1536: at every other cell a crystal structure already scores the fold, and a
# crystal outranks a GPU re-run of the same model. What a crystal cannot give is the model's own
# seed-to-seed spread, which is the floor a TT deviation is quoted beside. So this folds the 19
# missing 3abq_1536 cells (10 models x 2 seeds, less boltz2 s0 which is already banked) and
# nothing else.
#
# The box is instance 52151627, stopped rather than destroyed on 2026-09-23, so its ten upstream
# venvs and /root/ckpt survive; a fresh box would spend the whole budget on ref_setup.sh. The
# script starts it, folds, pulls the records back and stops it again on EVERY exit path, including
# a timeout or a Ctrl-C. vast_guard.sh is the backstop, not the plan: declare the hours in
# state/vast-budget before running this.
set -uo pipefail
cd "$(dirname "$0")/../../.."

IID=${MGX_IID:-52151627}; export IID
MAX_MIN=${MGX_MAX_MIN:-30}            # folding wall-clock cap, box time is this plus ~8 min
OUTDIR=${MGX_LOCAL_OUT:-/tmp/mgx_refs}
KEY=${MGX_SSH_KEY:-$HOME/.ssh/id_ed25519}
API=https://console.vast.ai/api/v0
K=$(tr -d '[:space:]' < "$HOME/.config/vastai/vast_api_key")
api() { curl -s -H "Authorization: Bearer $K" "$@"; }
log() { echo "[$(date -u +%FT%TZ)] $*"; }

# One worker per GPU, models CHEAPEST FIRST. The whole set is ~3.4 GPU-hours at 1536 (the 1280
# walls scaled by (1536/1280)^3 for the triangle ops), which does not fit the 1.5 h this row was
# allocated, so the run is designed to be cut off: cheapest first means the clock takes the
# expensive tail and every model it did reach has BOTH seeds, which is what a floor needs. Two
# workers rather than four because packing two folds onto one card does not add FLOPs, it only
# breaks the ordering that makes a cut-off run useful. ref_batch skips cells already ok, so a
# later pass resumes at the tail.
W0="boltz2 openbind esmfold2-fast opendde-abag protenix-v1"
W1="openfold3 rf3 opendde esmfold2 protenix-v2"

stop_box() {
  log "stopping instance $IID"
  api --request PUT "$API/instances/$IID/" \
      -H 'Content-Type: application/json' -d '{"state":"stopped"}' | head -c 300; echo
  sleep 20
  api "$API/instances/" | python3 -c '
import json,sys,os
iid=int(os.environ["IID"])
for i in json.load(sys.stdin).get("instances",[]):
    if i.get("id")==iid: print("instance", iid, "actual_status", i.get("actual_status"), "intended", i.get("intended_status"))
'
}
trap stop_box EXIT INT TERM

log "starting instance $IID"
api --request PUT "$API/instances/$IID/" -H 'Content-Type: application/json' \
    -d '{"state":"running"}' | head -c 300; echo

HOST=""; PORT=""
for _ in $(seq 1 60); do
  read -r st HOST PORT < <(api "$API/instances/" | python3 -c '
import json,sys,os
iid=int(os.environ["IID"])
for i in json.load(sys.stdin).get("instances",[]):
    if i.get("id")==iid:
        print(i.get("actual_status"), i.get("ssh_host") or "-", i.get("ssh_port") or "-")
')
  log "status=$st ssh=$HOST:$PORT"
  [ "$st" = running ] && break
  sleep 15
done
[ "${st:-}" = running ] || { log "instance never reached running"; exit 1; }

SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
     -o ConnectTimeout=10 -o ServerAliveInterval=30 root@$HOST"
for _ in $(seq 1 40); do
  $SSH true 2>/dev/null && break
  log "waiting for sshd"; sleep 15
done
$SSH true 2>/dev/null || { log "ssh never came up"; exit 1; }

REPO=$($SSH 'ls -d /root/tt-bio 2>/dev/null || ls -d /root/*/tt-bio 2>/dev/null | head -1')
[ -n "$REPO" ] || { log "no tt-bio checkout on the box"; exit 1; }
log "repo on box: $REPO"

# The box's copy of the tooling predates the stop, so ship the current one rather than pulling
# (the box has no credentials for a private remote). Fixtures are 19 MB.
rsync -az --delete -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  --exclude refs/ --exclude __pycache__/ --exclude .msa_tmp_env/ \
  perf/mgx/ref/ "root@$HOST:$REPO/perf/mgx/ref/" || { log "rsync out failed"; exit 1; }

log "launching 2 workers (one per GPU) on 3abq_1536, cap ${MAX_MIN} min"
$SSH "mkdir -p /root/results && cd $REPO && \
  setsid nohup bash -c '
    CUDA_VISIBLE_DEVICES=0 timeout ${MAX_MIN}m bash perf/mgx/ref/ref_batch.sh \"$W0\" 3abq_1536 \"0 1\" > /root/results/f_w0.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 timeout ${MAX_MIN}m bash perf/mgx/ref/ref_batch.sh \"$W1\" 3abq_1536 \"0 1\" > /root/results/f_w1.log 2>&1 &
    wait; echo FLOOR_DONE \$(date -u +%FT%TZ)
  ' > /root/results/floor.log 2>&1 < /dev/null &
  sleep 2; echo launched"

END=$(( $(date +%s) + MAX_MIN*60 + 120 ))
while [ "$(date +%s)" -lt "$END" ]; do
  sleep 120
  n=$($SSH "ls /root/refs/*/3abq_1536/s*.json 2>/dev/null | wc -l" 2>/dev/null || echo '?')
  ok=$($SSH "grep -l '\"status\": \"ok\"' /root/refs/*/3abq_1536/s*.json 2>/dev/null | wc -l" 2>/dev/null || echo '?')
  log "records=$n ok=$ok"
  $SSH "grep -q FLOOR_DONE /root/results/floor.log" 2>/dev/null && { log "workers finished"; break; }
done

mkdir -p "$OUTDIR"
rsync -az -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  "root@$HOST:/root/refs/" "$OUTDIR/" && log "records pulled to $OUTDIR"
rsync -az -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  "root@$HOST:/root/results/" "$OUTDIR/../mgx_floor_logs/" 2>/dev/null
log "done folding; EXIT trap stops the box"

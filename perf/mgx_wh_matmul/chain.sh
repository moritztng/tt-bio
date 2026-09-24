#!/bin/bash
# Detached probe chain on whglx: wait for a cleanly released chip, run each line of JOBS (a script
# path plus its args) whose output file (the value after --out) does not exist yet, release it.
# Usage (from the whglx worktree):
#   setsid nohup perf/mgx_wh_matmul/chain.sh perf/mgx_wh_matmul/jobs/<name>.txt > <log> 2>&1 < /dev/null &
cd "$(dirname "$0")/../.." || exit 1
JOBS=$1
sleep 20000 & HOLD=$!
CARD=$(timeout 10800 python3 perf/mgx_wh_matmul/claim.py $HOLD) || { echo "no chip in 3 h"; kill $HOLD; exit 1; }
echo "$(date -u +%FT%TZ) claimed card $CARD (hold pid $HOLD)"
export CARD HOLD_PID=$HOLD
while read -r line; do
    [ -z "$line" ] && continue
    out=$(sed -n 's/.*--out \([^ ]*\).*/\1/p' <<<"$line")
    [ -n "$out" ] && [ -s "$out" ] && { echo "skip $line"; continue; }
    echo "$(date -u +%FT%TZ) start $line"
    # shellcheck disable=SC2086
    timeout ${LINE_TIMEOUT:-9000} perf/mgx_wh_matmul/run.sh $line
    echo "$(date -u +%FT%TZ) rc=$? $line"
done < "$JOBS"
python3 - "$CARD" <<'PY'
import json, sys, time
p = f"/home/agent/leases/j10glx02-card{sys.argv[1]}.json"
d = json.load(open(p))
if d.get("holder") == "worker:mgx-wh-matmul":
    d.update(released=time.time(), note="released by mgx-wh-matmul chain")
    json.dump(d, open(p, "w"))
PY
kill $HOLD
echo "$(date -u +%FT%TZ) chain finished, card $CARD released"

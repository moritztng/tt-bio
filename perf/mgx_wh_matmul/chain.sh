#!/bin/bash
# Detached probe chain on whglx: wait for a cleanly released chip, run the probes, release it.
# Usage (from the whglx worktree): nohup perf/mgx_wh_matmul/chain.sh > perf/mgx_wh_matmul/results/chain.log 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
R=perf/mgx_wh_matmul/results
mkdir -p $R
sleep 20000 & HOLD=$!
CARD=$(timeout 10800 python3 perf/mgx_wh_matmul/claim.py $HOLD) || { echo "no chip in 3 h"; kill $HOLD; exit 1; }
echo "$(date -u +%FT%TZ) claimed card $CARD (hold pid $HOLD)"
export CARD HOLD_PID=$HOLD
run() { echo "$(date -u +%FT%TZ) start $*"; timeout 1500 perf/mgx_wh_matmul/run.sh "$@"; echo "$(date -u +%FT%TZ) rc=$? $1"; }
[ -s $R/knobs.json ] || run perf/mgx_wh_matmul/knobs.py --out $R/knobs.json
[ -s $R/element_kt16.json ] || run perf/mgx_wh_matmul/element.py --kt 16 --out $R/element_kt16.json
[ -s $R/element_kt64.json ] || run perf/mgx_wh_matmul/element.py --kt 64 --max-elems 3 --out $R/element_kt64.json
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

#!/bin/bash
# Run one probe pinned to CARD (default 31) on whglx, then re-claim the lease so a sibling row does
# not take the chip between probes. Usage: CARD=31 perf/mgx_wh_matmul/run.sh script.py args...
CARD=${CARD:-31}
cd "$(dirname "$0")/../.." || exit 1
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:mgx-wh-matmul \
    ~/env/bin/python3 "$@" 2>&1 | grep -v -E "DEBUG|\| INFO|^Config|device bring-up lock"
rc=${PIPESTATUS[0]}
python3 - "$CARD" <<'PY'
import json, os, sys, time
c = sys.argv[1]
p = f"{os.path.expanduser('~')}/leases/j10glx02-card{c}.json"
d = json.load(open(p))
if d.get("holder") == "worker:mgx-wh-matmul":
    d.update(released=None, pid=int(os.environ.get("HOLD_PID", os.getppid())), note="held between probes by mgx-wh-matmul")
    json.dump(d, open(p, "w"))
PY
exit $rc

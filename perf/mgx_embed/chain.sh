#!/bin/bash
# Run perf/mgx_embed jobs one after another on one whglx chip, holding the chip's lease file
# between them (tt_bio's own lease lives only as long as the process that opens the device).
#   bash perf/mgx_embed/chain.sh <jobs.txt> [preferred card]
# Each line of jobs.txt is a command run from the clone root; `#` lines are skipped.
#
# A chip is taken only if its lease is ours, its holder pid is dead, or it was released at least
# QUIET_S ago: rows on this box release between rungs and re-claim seconds later, so a fresh
# `released` is someone else's chip mid-walk. Cards 1 and 24-27 are never candidates.
jobs=$1; card=${2:-}
wt=$(cd "$(dirname "$0")/../.." && pwd)
export TT_BIO_LEASE_DIR=/home/agent/leases TT_BIO_LEASE_HOLDER=worker:mgx-embed-scale \
       TT_BIO_LEASE_TIMEOUT=600 ESM_ROOT=${ESM_ROOT:-/home/mthuening/work/esm} \
       OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
S=$HOME/scratch/mgxembed; export S
claim() {  # prints the claimed card, exit 1 if none is free
  python3 - "$1" "$2" "${QUIET_S:-120}" <<'EOF'
import glob, json, os, sys, time
pid, pref, quiet = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
ME, BLOCKED = "worker:mgx-embed-scale", {"1", "24", "25", "26", "27"}
def alive(p):
    try:
        os.kill(int(p), 0); return True
    except (OSError, TypeError, ValueError):
        return False
cards = [f.split("card")[1][:-5] for f in sorted(glob.glob("/home/agent/leases/j10glx02-card*.json"))]
for c in ([pref] if pref else []) + cards:
    if c in BLOCKED:
        continue
    p = f"/home/agent/leases/j10glx02-card{c}.json"
    m = json.load(open(p)) if os.path.exists(p) else {}
    rel = m.get("released")
    ok = (not m or (m.get("holder") == ME and (m.get("pid") == pid or not alive(m.get("pid"))))
          or (rel and time.time() - rel > quiet) or (not rel and not alive(m.get("pid"))))
    if ok:
        json.dump({"host": "j10glx02", "card": c, "holder": ME, "pid": pid,
                   "acquired": time.time(), "released": None}, open(p, "w"))
        print(c); sys.exit(0)
sys.exit(1)
EOF
}
release() {
  python3 - "$1" "$2" <<'EOF'
import json, sys, time
p = f"/home/agent/leases/j10glx02-card{sys.argv[1]}.json"
m = json.load(open(p))
# Only our own claim: another row may have taken the chip after our last job.
if m.get("holder") == "worker:mgx-embed-scale" and m.get("pid") == int(sys.argv[2]) and not m.get("released"):
    m["released"] = time.time(); json.dump(m, open(p, "w"))
EOF
}
cd "$wt"
while IFS= read -r cmd; do
  [[ -z "$cmd" || "$cmd" == \#* ]] && continue
  until card=$(claim $$ "$card"); do sleep 60; done
  export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card
  echo "=== $(date -u +%FT%TZ) card $card: $cmd"
  eval "$cmd" < /dev/null
  echo "=== rc=$? $(date -u +%FT%TZ)"
done < "$jobs"
[ -n "$card" ] && release "$card" $$
echo "=== chain done $(date -u +%FT%TZ)"

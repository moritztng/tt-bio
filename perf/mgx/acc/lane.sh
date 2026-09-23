#!/bin/bash
# One device lane on whglx: take jobs from a shared queue file, one fold at a time, each on a chip
# whose lease is absent or released, under our own lease for the whole fold.
#
#   perf/mgx/acc/lane.sh JOBS OUT
#
# JOBS lines: "<tag> <tree> <model> <fixture>"; a job is claimed by appending its tag to
# JOBS.claimed under flock, so two lanes never run the same one. Each finished job appends one
# row to OUT/results.jsonl (ladder.py's row plus tree commit and the DURING-sampled AICLK).
# Cards 1 and 24-27 carry the live app and a co-tenant and are never taken.
set -u
JOBS=$(realpath "$1"); OUT=$(realpath -m "$2")
LEASES=/home/agent/leases; HOST=j10glx02; HOLDER=worker:mgx-accuracy
PY=/home/agent/env/bin/python
mkdir -p "$OUT"

free_card() {  # the lease is the flock (tt_bio/device_lease.py); a record alone can be stale
  for c in $(seq 0 31); do
    case $c in 1|24|25|26|27) continue;; esac
    f=$LEASES/$HOST-card$c.json
    if [ ! -s "$f" ] || python3 -c "import json,sys; sys.exit(0 if json.load(open('$f')).get('released') else 1)" 2>/dev/null; then
      flock -n "$f" true 2>/dev/null && { echo $c; return 0; }
    fi
  done
  return 1
}

lease() {  # card released(null|epoch); writes only under the card's free flock, releases only our own record
  flock -n "$LEASES/$HOST-card$1.json" python3 - "$LEASES/$HOST-card$1.json" "$1" "$2" $$ <<'EOF'
import json, sys, time
p, card, rel, pid = sys.argv[1:]
try:
    cur = json.load(open(p))
except Exception:
    cur = {}
if rel != "null" and cur.get("holder") not in (None, "worker:mgx-accuracy"):
    sys.exit(0)  # another worker took the card after our fold ended
json.dump({"host": "j10glx02", "card": card, "holder": "worker:mgx-accuracy", "pid": int(pid),
           "acquired": time.time(), "released": None if rel == "null" else time.time()}, open(p, "w"))
EOF
}

claim() {
  exec 9>>"$JOBS.claimed"
  flock 9
  while read -r tag tree model fx; do
    [ -z "$tag" ] || [ "${tag:0:1}" = "#" ] && continue
    grep -qx "$tag" "$JOBS.claimed" && continue
    echo "$tag" >> "$JOBS.claimed"
    flock -u 9
    echo "$tag $tree $model $fx"
    return 0
  done < "$JOBS"
  flock -u 9
  return 1
}

while job=$(claim); do
  read -r tag tree model fx <<<"$job"
  exec 8>>"$OUT/.cardlock"
  flock 8
  until card=$(free_card); do flock -u 8; sleep 60; flock 8; done
  lease "$card" null
  flock -u 8
  commit=$(git -C "$tree" rev-parse --short=9 HEAD)
  ( while :; do
      v=$(TT_VISIBLE_DEVICES=$card timeout 30 /usr/local/bin/tt-smi -s 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['device_info'][0]['telemetry']['aiclk'])" 2>/dev/null)
      [ -n "$v" ] && echo "$(date +%s) $v"
      sleep 30
    done ) > "$OUT/$tag.aiclk" 2>/dev/null &
  clk=$!
  (cd "$tree" && PYTHONPATH="$tree" $PY perf/whceil/ladder.py --model "$model" --device "$card" \
     --rungs "$tree/perf/mgx/ref/fixtures/$fx.yaml" \
     --env TT_BIO_SIZE_LIMIT=0 --env TT_BIO_LEASE_DIR=$LEASES --env TT_BIO_LEASE_CARDS=$card \
     --env TT_BIO_LEASE_HOLDER=$HOLDER --env TT_METAL_CACHE=/home/agent/.cache/tt-metal-cache-mgxacc \
     --out "$OUT/$tag.jsonl" --out-root "$OUT/runs/$tag" --timeout 5400 -- --host_threads 2) \
     > "$OUT/$tag.driver.log" 2>&1
  kill $clk 2>/dev/null
  lease "$card" now
  $PY - "$OUT" "$tag" "$tree" "$commit" "$card" <<'EOF'
import json, statistics, sys
out, tag, tree, commit, card = sys.argv[1:]
try:
    row = json.loads(open(f"{out}/{tag}.jsonl").read().strip().splitlines()[-1])
except Exception as e:
    row = {"verdict": "NO_ROW", "error": str(e)}
clk = [int(l.split()[1]) for l in open(f"{out}/{tag}.aiclk") if l.split()[1:2] and l.split()[1].isdigit()]
row.update(tag=tag, tree=tree, commit=commit, card=int(card),
           aiclk_mhz={"n": len(clk), "min": min(clk), "median": statistics.median(clk), "max": max(clk)} if clk else None)
open(f"{out}/results.jsonl", "a").write(json.dumps(row) + "\n")
sys.exit(3 if row.get("rc") == 75 else 0)
EOF
  # rc 75 = another process won the card between our pick and the fold's open; nothing ran.
  [ $? = 3 ] && [ "${tag%+++}" = "$tag" ] && echo "$tag+ $tree $model $fx" >> "$JOBS"
done

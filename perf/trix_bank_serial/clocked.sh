#!/usr/bin/env bash
# clocked.sh <node> <label> -- <command...>
# Runs the command with a ~500 Hz AICLK sampler on <node> alive for exactly its duration, so the
# clock is sampled DURING the measurement and not before it. Prints the sampled distribution.
set -u
NODE="${1:?node}"; LABEL="${2:?label}"; shift 2
[ "${1:-}" = "--" ] && shift
HERE="$(cd "$(dirname "$0")" && pwd)"
CLK="$HERE/out/aiclk_${LABEL}.jsonl"
python3 - "$NODE" "$CLK" <<"PY" &
import sys, time, json
node, out = sys.argv[1], sys.argv[2]
src = "/sys/class/tenstorrent/tenstorrent!%s/tt_aiclk" % node
with open(out, "w") as f:
    while True:
        try:
            f.write(json.dumps({"t": time.time(), "MHz": int(open(src).read())}) + "\n")
        except Exception as e:
            f.write(json.dumps({"t": time.time(), "err": repr(e)}) + "\n")
        f.flush()
        time.sleep(0.002)
PY
SPID=$!
"$@"; rc=$?
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
python3 - "$CLK" <<"PY"
import json, sys, statistics as st
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
mhz = [r["MHz"] for r in rows if "MHz" in r]
if not mhz:
    print("CLOCK: no samples"); raise SystemExit
q = sorted(mhz)
print("CLOCK n=%d min=%d p05=%d median=%d p95=%d max=%d mean=%.1f  >=1200: %.1f%%" % (
    len(q), q[0], q[int(.05*len(q))], st.median(q), q[int(.95*len(q))-1], q[-1],
    st.fmean(q), 100.0*sum(v >= 1200 for v in q)/len(q)))
PY
exit $rc

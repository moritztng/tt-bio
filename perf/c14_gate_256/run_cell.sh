#!/usr/bin/env bash
# run_cell.sh <tag> <rung> <processes> <loadN>
#
# One cell of the 256-rung mode discriminator. Invoke it THROUGH benchlock:
#
#   benchlock.sh worker:c14-gate-256-warmup -- bash run_cell.sh q256 256 3 0
#
# <processes> FRESH processes, each folding the rung's fixture FOLDS (default 5) times back to
# back via fold_seq.py. The AICLK is forced to 1350 on this card's sysfs node and sampled at
# ~500 Hz for exactly the duration of the folds, so the clock is recorded DURING.
#
# loadN > 0 starts that many host CPU burners for the cell and kills them after: the LOADED
# condition. The burners never open a device, so the only channel they share with the fold is
# the host. benchlock still keeps every OTHER worker's timed run out of the window.
#
# qb1's node map is permuted: UMD 3 (this row's card grant) is /dev/tenstorrent/0.
set -u
TAG="${1:?tag}"; RUNG="${2:?rung}"; NP="${3:-3}"; LOADN="${4:-0}"
WT=/home/ttuser/.coworker/wt/c14-gate-256-warmup
D=$WT/perf/c14_gate_256
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=3
NODE=0
FOLDS=${FOLDS:-5}
cd "$WT" || exit 2
mkdir -p "$D/out"

echo "=== cell $TAG rung=$RUNG processes=$NP folds=$FOLDS loadN=$LOADN  $(date -u +%FT%TZ)"
echo "host_quiet at cell start:"
$PY "$WT/perf/c12_orchestrator/pair_guard/host_quiet.py" || true
echo "loadavg at cell start: $(cat /proc/loadavg)"

BURN=""
if [ "$LOADN" -gt 0 ]; then
  for _ in $(seq "$LOADN"); do
    $PY -c 'import time
t = time.time(); x = 0
while time.time() - t < 3600:
    x = (x * 1103515245 + 12345) & 0xFFFFFFFF' &
    BURN="$BURN $!"
  done
  sleep 25
  echo "burners:$BURN"
  echo "loadavg after burner spin-up: $(cat /proc/loadavg)"
fi

CLK="$D/out/aiclk_${TAG}.jsonl"
$PY - "$NODE" "$CLK" <<'PY' &
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

# 1 Hz host-load + foreign-device-holder stamp, so every fold can be read back against the box
# it actually ran on. Pass 1 could only stamp load per PROCESS, which cannot tell a fold slowed
# by a co-tenant from one slowed by warm-up.
LOADLOG="$D/out/load_${TAG}.jsonl"
$PY - "$LOADLOG" <<PY &
import json, os, sys, time
out = sys.argv[1]
def holders():
    n = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or pid == str(os.getpid()):
            continue
        fd = "/proc/%s/fd" % pid
        try:
            for f in os.listdir(fd):
                if "tenstorrent" in os.readlink(os.path.join(fd, f)):
                    n += 1
                    break
        except OSError:
            pass
    return n
with open(out, "w") as f:
    while True:
        la = os.getloadavg()
        f.write(json.dumps({"t": time.time(), "load1": la[0], "dev": holders()}) + "\n")
        f.flush()
        time.sleep(1.0)
PY
LPID=$!

$PY "$WT/perf/c10_core_grid/force_aiclk.py" "$NODE" 1350 36000 \
    > "$D/out/force_${TAG}.log" 2>&1 &
FPID=$!
sleep 3
head -2 "$D/out/force_${TAG}.log"

rc=0
i=0
while [ "$i" -lt "$NP" ]; do
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
      TT_BIO_LEASE_HOLDER=worker:c14-gate-256-warmup \
      "$PY" "$D/fold_seq.py" --rung "$RUNG" --folds "$FOLDS" \
            --tag "${TAG}p${i}" --out "$D/out/${TAG}p${i}.json" || rc=$?
  i=$((i + 1))
done

kill "$FPID" 2>/dev/null; wait "$FPID" 2>/dev/null
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
kill "$LPID" 2>/dev/null; wait "$LPID" 2>/dev/null
for p in $BURN; do kill "$p" 2>/dev/null; done
for p in $BURN; do wait "$p" 2>/dev/null; done

$PY - "$CLK" <<'PY'
import json, sys, statistics as st
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
mhz = [r["MHz"] for r in rows if "MHz" in r]
if not mhz:
    print("CLOCK: no samples"); raise SystemExit
q = sorted(mhz)
print("CLOCK n=%d min=%d p05=%d median=%d p95=%d max=%d mean=%.1f  >=1200: %.1f%%" % (
    len(q), q[0], q[int(.05 * len(q))], st.median(q), q[int(.95 * len(q)) - 1], q[-1],
    st.fmean(q), 100.0 * sum(v >= 1200 for v in q) / len(q)))
PY
echo "loadavg at cell end: $(cat /proc/loadavg)"
echo "=== cell $TAG rc=$rc  $(date -u +%FT%TZ)"
exit $rc

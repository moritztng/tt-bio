#!/usr/bin/env bash
# The measured half of run.sh: runs with the benchlock already held.
set -u
WT=/home/ttuser/.coworker/wt/pvx-protenix-specific
PY=/home/ttuser/tt-bio-dev/env/bin/python3      # the venv every other pvx row uses (ttnn 0.68.0);
                                                # /usr/bin/python3 here carries 0.67.4 and a row on
                                                # a different ttnn is not comparable to its siblings
NODE="${PVX_CLK_NODE:-0}"; CARD="${PVX_CARD:-3}"
TAG="$PVX_TAG"; MODEL="$PVX_MODEL"
cd "$WT" || exit 1

$PY perf/pvx_ps/pin_aiclk.py "$NODE" 1350 > "perf/pvx_ps/${TAG}.pin.log" 2>&1 &
PIN=$!
for _ in $(seq 1 20); do grep -q READY "perf/pvx_ps/${TAG}.pin.log" 2>/dev/null && break; sleep 0.5; done

TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
TT_BIO_LEASE_HOLDER="worker:pvx-protenix-specific" PYTHONPATH="$WT" \
  $PY -u perf/pvx_ps/census.py --model "$MODEL" \
      --out "$WT/perf/pvx_ps/${TAG}.json" $PVX_ARGS > "perf/pvx_ps/${TAG}.log" 2>&1
RC=$?
kill -TERM "$PIN" 2>/dev/null; wait "$PIN" 2>/dev/null
echo "census rc=$RC" >> "perf/pvx_ps/${TAG}.log"
exit $RC

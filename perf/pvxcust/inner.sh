#!/usr/bin/env bash
# The measured half of run.sh: runs with the benchlock already held.
# Card UMD 0 is PCI 0000:01:00.0, which is /dev/tenstorrent node 1 on this box -- UMD id and
# device node are NOT the same number here, so the clock pin and the compute grant are set from
# two different variables on purpose.
set -u
WT=/home/ttuser/.coworker/wt/pvx-custchart
PY=/home/ttuser/tt-bio-dev/env/bin/python3   # the venv every pvx row uses (ttnn 0.68.0);
                                             # /usr/bin/python3 here carries 0.67.4
NODE="${PVXC_CLK_NODE:-1}"; CARD="${PVXC_CARD:-0}"; PIN="${PVXC_PIN:-1350}"
TAG="$PVXC_TAG"; LEGS="$PVXC_LEGS"
cd "$WT" || exit 1

if [ "$PIN" != "0" ]; then
  $PY perf/pvxcust/pin_aiclk.py "$NODE" "$PIN" > "perf/pvxcust/${TAG}.pin.log" 2>&1 &
  PINPID=$!
  for _ in $(seq 1 20); do grep -q READY "perf/pvxcust/${TAG}.pin.log" 2>/dev/null && break; sleep 0.5; done
else
  PINPID=""
fi

PVXC_COMMIT="$(git rev-parse HEAD)" PVXC_PIN="$PIN" \
TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
TT_BIO_LEASE_HOLDER="worker:pvx-custchart" PYTHONPATH="$WT" \
  $PY -u perf/pvxcust/fold_cust.py --legs "$LEGS" \
      --out "$WT/perf/pvxcust/${TAG}.json" > "perf/pvxcust/${TAG}.log" 2>&1
RC=$?
[ -n "$PINPID" ] && { kill -TERM "$PINPID" 2>/dev/null; wait "$PINPID" 2>/dev/null; }
echo "fold_cust rc=$RC" >> "perf/pvxcust/${TAG}.log"
exit $RC

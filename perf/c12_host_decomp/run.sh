#!/usr/bin/env bash
# One benchlocked capture per size, one process and one device context each.
#
#   bash perf/c12_host_decomp/run.sh <name> [node] [sizes...]
#
# The node argument is not cosmetic: qb2's live card moved when the box rebooted on 2026-09-17,
# and the row's card grant has to be checked against sysfs rather than assumed.
set -euo pipefail
NAME="${1:?usage: run.sh <name> [node] [sizes...]}"; shift || true
NODE="${1:-0}"; shift || true
SIZES=("$@"); [ ${#SIZES[@]} -gt 0 ] || SIZES=(512 298)
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ARMS="${ARMS:-bare,regions,cacheclear,cprofile,sample,pyspy}"
REPS="${REPS:-3}"
CLOCKS="${CLOCKS:-1350,800}"
# The row running the capture. It names the benchlock slug, the device lease holder and the
# holder decomp.py checks for, so a second row can reuse this harness without forking it.
HOLDER="${HOLDER:-worker:c12-host-decomp}"
SLUG="${HOLDER#worker:}"

# Not `python3`: a non-interactive ssh shell does not source the venv carrying torch + ttnn
# 0.68.0, so a bare python3 dies on `import torch` after benchlock has already been taken.
# Resolve the interpreter explicitly and prove it imports before spending a lock on it.
PY="${PY:-/home/ttuser/tt-bio-dev/env/bin/python3}"
if ! "$PY" -c "import torch, ttnn" >/dev/null 2>&1; then
  echo "run.sh: $PY cannot import torch+ttnn. Set PY=<venv>/bin/python3. Refusing." >&2
  exit 1
fi

# Existence is not readability. A quarantined or ARC-dead chip KEEPS this sysfs file and fails
# only the READ, with ENODATA, so `[ -r ... ]` passed a dead chip through on 2026-09-17 and the
# capture then host-spun on it. Read the value.
if ! cat "/sys/class/tenstorrent/tenstorrent!${NODE}/tt_aiclk" >/dev/null 2>&1; then
  echo "run.sh: node ${NODE} tt_aiclk exists but cannot be READ -- ARC dead or chip quarantined." >&2
  echo "run.sh: that state needs a REBOOT, not a reset. Refusing." >&2
  exit 1
fi

# Two guards, both added after pass 3 lost its whole turn on this box. Neither is optional and
# both are cheap relative to a capture.
#
# 1. The sibling chip. A p300c's two chips share one board power budget, so a fold on the sibling
#    moves this node's timing while never touching benchlock's lock file: benchlock's contract is
#    mutual exclusion among benchlock CALLERS, and a job that never asks for the lock is
#    structurally invisible to it. `pair_idle.py` is c12-orchestrator's tool and is READ FROM ITS
#    BRANCH rather than copied here, so the two rows cannot drift apart.
# 2. Whether this chip can run a program at all. On 2026-09-17 node 2 accepted the 32x32 bf16 add
#    that tt_bio's own bring-up probe issues and never completed it, and because that probe runs
#    inline in the capture process there was no deadline on it: twelve minutes, no fold, no rows.
#    `dispatch_probe.py` runs the same test in a child under a timeout, so a wedged chip costs the
#    timeout instead of the pass.
PAIR_IDLE="$ROOT/perf/c12_orchestrator/pair_guard/pair_idle.py"
if ! [ -r "$PAIR_IDLE" ]; then
  PAIR_IDLE="$(mktemp -t pair_idle.XXXXXX.py)"
  if ! git -C "$ROOT" show origin/wk/c12-orchestrator:perf/c12_orchestrator/pair_guard/pair_idle.py \
       > "$PAIR_IDLE" 2>/dev/null; then
    echo "run.sh: cannot reach pair_idle.py on wk/c12-orchestrator. Fetch it, do not skip it." >&2
    exit 1
  fi
fi
if ! "$PY" "$PAIR_IDLE" --card "$NODE"; then
  echo "run.sh: sibling chip of node ${NODE} is busy -- a p300c pair is ONE timing resource." >&2
  echo "run.sh: wait it out or DEFER. Refusing to measure and explain it afterwards." >&2
  exit 75
fi

# The dispatch probe OPENS the device, so it has to run INSIDE benchlock, not before it. Run
# before, it collided with a co-tenant row's live timed fold on the same card on 2026-09-18: the
# port read is free, but the 32x32 add is a second opener on a chip somebody else is timing on.
# It is launched with the capture below instead.

for size in "${SIZES[@]}"; do
  out="$HERE/runs/$NAME/$size"
  mkdir -p "$HERE/runs/$NAME"
  echo "=== $size aa -> $out (node $NODE, clocks $CLOCKS, arms $ARMS, reps $REPS) ==="
  # One size failing must not cost the other size or the table: decomp.py saves every row as
  # it lands, so a partial result.json is still evidence and fit.py reads only valid rows.
  rc=0
  TT_VISIBLE_DEVICES="$NODE" TT_BIO_LEASE_CARDS="$NODE" \
  TT_BIO_LEASE_HOLDER="$HOLDER" \
  ~/.coworker/scripts/benchlock.sh "$SLUG" -- \
    bash -c '"$0" "$1" --node "$2" --timeout "$3" && "$0" "$4" --size "$5" --node "$2" \
             --out "$6" --clocks "$7" --reps "$8" --arms "$9" --holder "${10}"' \
      "$PY" "$HERE/dispatch_probe.py" "$NODE" "${PROBE_TIMEOUT_S:-240}" "$HERE/decomp.py" \
      "$size" "$out" "$CLOCKS" "$REPS" "$ARMS" "$HOLDER" \
    > >(tee "$HERE/runs/$NAME/launch_$size.log") 2>&1 || rc=$?
  wait
  if [ "$rc" = 75 ]; then
    echo "run.sh: benchlock busy (75) on $size aa -- NOT measured, retry later." >&2
  elif [ "$rc" != 0 ]; then
    echo "run.sh: $size aa capture exited $rc -- keeping whatever rows landed." >&2
  fi
done

"$PY" "$HERE/fit.py" "$HERE"/runs/"$NAME"/*/result.json > "$HERE/runs/$NAME/table.json"
echo "table: $HERE/runs/$NAME/table.json"

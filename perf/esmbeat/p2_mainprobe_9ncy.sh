#!/usr/bin/env bash
# Does boltz2-9ncy-nomsa already GAP on origin/main, with none of this branch's commits present?
# Throwaway detached worktree under this slug's own namespace, removed at the end.
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-p2
PROBE=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-p2-mainprobe
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
git worktree remove --force "$PROBE" 2>/dev/null
git fetch -q origin
git worktree add --detach "$PROBE" origin/main || exit 1
cd "$PROBE" || exit 1
echo "probe HEAD: $(git rev-parse --short HEAD)  vs origin/main: $(git rev-parse --short origin/main)"
bash scripts/fetch_parity_fixtures.sh >/dev/null 2>&1 || echo "FIXTURE FETCH FAILED"
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200-p2
export PYTHONPATH="$PROBE"
export ESM_ROOT=/home/ttuser/esm
$PY -u scripts/full_parity_gate.py --leg boltz2-9ncy-nomsa \
  --workers localhost:0 --workdir "$PROBE/.probe" 2>&1 | grep -E "^boltz2-9ncy|Tally"
cp "$PROBE/.probe/boltz2-9ncy-nomsa.json" "$WT/perf/esmbeat/iso_9ncy_originmain.json" 2>/dev/null
cd "$WT" || exit 1
git worktree remove --force "$PROBE"
echo "PROBE_DONE, worktree removed: $(test -d "$PROBE" && echo NO || echo yes)"

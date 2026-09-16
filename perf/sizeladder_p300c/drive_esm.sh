#!/bin/bash
# Record esmfold2's missing p300c rungs, ONE rung per process, ascending, 512 last.
#
# qb2 watchdog-resets the box roughly every 10 minutes while a fold is running (bootstatus=32,
# five resets in this campaign: 08:52, 09:26, 09:38, 09:52, 10:29 UTC), and a record pass only
# flushes after the whole model completes, so a multi-rung slice loses everything it measured.
# One rung per process keeps each flush inside the window; the recorder's own carry folds the
# rungs this process did not measure in from the previous entry (same commit, host and grid),
# so the entry grows a rung at a time and nothing stale comes back.
#
# Idempotent: re-running skips rungs already recorded, so a relaunch after a reset costs only
# the rungs that were still in flight. 512 goes last and is re-run whenever sigma is missing:
# it is the sigma rung, and a single-rung slice writes no exponent block, so sigma only lands
# on a pass where 512 is measured AND the other rungs are already there to exponent over.
WT=/home/ttuser/.coworker/wt/tt-bio-sizeladder-p300c-refresh
cd "$WT" || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
FRAG=docs/size_ladder_baseline.d/esmfold2.json

todo=$("$PY" - <<'PYEOF'
import json
e = json.load(open("docs/size_ladder_baseline.d/esmfold2.json"))["cards"]["p300c"]["models"]["esmfold2"]
have = set(e.get("runtime_s") or {}) | set(e.get("refused") or {})
todo = [r for r in ("640", "768", "896", "1024") if r not in have]
if e.get("sigma_runtime_512") is None or "512" not in have:
    todo.append("512")   # last: sigma needs the rest of the ladder present to exponent over
print(" ".join(todo))
PYEOF
)
echo "[drive] $(date -u +%H:%M:%SZ) rungs to record: ${todo:-none}"
for r in $todo; do
  echo "[drive] $(date -u +%H:%M:%SZ) rung $r starting"
  RELEASE_GATE_SIZE_RUNGS=$r TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
    TT_BIO_LEASE_HOLDER=worker:tt-bio-sizeladder-p300c-refresh \
    "$PY" -u scripts/release_gate.py --model size-ladder --size-ladder-record \
      --size-ladder-fragment --size-ladder-models esmfold2 --keep \
      > perf/sizeladder_p300c/rec_esm_r$r.log 2>&1
  echo "[drive] $(date -u +%H:%M:%SZ) rung $r exit $? -- $(grep -c 'recorded to' perf/sizeladder_p300c/rec_esm_r$r.log) flush(es)"
done
echo "[drive] $(date -u +%H:%M:%SZ) done"

#!/bin/bash
# Same per-rung recorder as drive_esm.sh, 512 FIRST. The exponent block is gated over
# 256/512/768 only, and all three are recorded, so measuring 512 now lands sigma and the
# exponent gate even if the box resets before 1024 goes in. Six watchdog resets in this
# campaign, so the order the rungs land in is worth choosing.
WT=/home/ttuser/.coworker/wt/tt-bio-sizeladder-p300c-refresh
cd "$WT" || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
todo=$("$PY" - <<'PYEOF'
import json
e = json.load(open("docs/size_ladder_baseline.d/esmfold2.json"))["cards"]["p300c"]["models"]["esmfold2"]
have = set(e.get("runtime_s") or {}) | set(e.get("refused") or {})
todo = ["512"] if (e.get("sigma_runtime_512") is None or "512" not in have) else []
todo += [r for r in ("640", "768", "896", "1024") if r not in have]
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

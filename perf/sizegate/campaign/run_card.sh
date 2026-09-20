#!/bin/bash
# One card's slice of the p150a ladder re-record. Args: <card> <model> [model...]
#
# "Already recorded" is read off the ARTIFACT, not off a marker file: the fragment has to
# carry a cell at all four new rungs for this card. A .done touched by an exit code says the
# process ended, which is not the same claim.
#
# The claim is `mkdir`, which is atomic, so three cards walking overlapping lists never
# record the same model twice into the same fragment.
set -u
WT=/home/ttuser/.coworker/wt/cov-ladder-below-bar-all-bhp150a
PY=/home/ttuser/kisoji_p2_fresh/env/bin/python3
CARD=$1; shift
cd "$WT" || exit 1
mkdir -p perf/sizegate/campaign/logs perf/sizegate/campaign/claim

recorded() {
  "$PY" - "$1" <<'PYEOF'
import json, pathlib, sys
f = pathlib.Path("docs/size_ladder_baseline.d") / (sys.argv[1] + ".json")
NEW = {"1152", "1280", "1408", "1536"}
try:
    e = json.loads(f.read_text())["cards"]["p150a"]["models"][sys.argv[1]]
except Exception:
    sys.exit(1)
have = set(e.get("runtime_s") or {}) | set(e.get("refused") or {})
sys.exit(0 if NEW <= have else 1)
PYEOF
}

for M in "$@"; do
  if recorded "$M"; then echo "[card $CARD] $M already carries the new rungs"; continue; fi
  if ! mkdir "perf/sizegate/campaign/claim/$M" 2>/dev/null; then
    echo "[card $CARD] $M claimed by another card"; continue
  fi
  LOG=perf/sizegate/campaign/logs/$M.card$CARD.log
  echo "=== $M card $CARD start $(date -u +%FT%TZ) ===" >> "$LOG"
  PYTHONPATH="$WT" RELEASE_GATE_CENSUS_PYTHONPATH="$WT" \
  RELEASE_GATE_SIZE_WORKDIR="$WT/perf/sizegate/work-card$CARD" \
  TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=0,$CARD \
  TT_BIO_LEASE_HOLDER=worker:cov-ladder-below-bar-all-bhp150a \
    "$PY" scripts/release_gate.py --model size-ladder --size-ladder-record \
      --size-ladder-fragment --size-ladder-models "$M" --load-ceiling 0 >> "$LOG" 2>&1
  echo "=== $M card $CARD rc=$? $(date -u +%FT%TZ) ===" >> "$LOG"
  if recorded "$M"; then
    echo "=== $M card $CARD fragment carries all four new rungs ===" >> "$LOG"
  else
    # release the claim so another card can retry; the artifact is the judge
    rmdir "perf/sizegate/campaign/claim/$M" 2>/dev/null
  fi
done
echo "[card $CARD] slice finished $(date -u +%FT%TZ)"

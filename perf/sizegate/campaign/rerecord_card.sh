#!/bin/bash
# One card's slice of the p150a ladder re-record at THIS branch's engine. Args: <card> <model...>
#
# Why a re-record at all: every fragment in docs/size_ladder_baseline.d records
# SDPA_WIDE_K resolved "False", and the lever has shipped ON by default since 7662fc58b,
# which is on origin/main. A live fold at HEAD reads "True" and the arm's own comparator
# calls that "default or threshold constant changed" -- one finding per model per rung on
# every card. The arm's stated rule is to re-record after a size-affecting change, and the
# row that flipped the default did not.
#
# "Already done" is read off the ARTIFACT's engine, not off a marker and not off which
# rungs exist: a fragment recorded at a commit whose tt_bio differs from HEAD's is stale no
# matter how complete it is. The claim is mkdir, which is atomic, so two cards walking
# overlapping lists never record one model twice.
set -u
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY=/home/ttuser/kisoji_p2_fresh/env/bin/python3
CARD=$1; shift
cd "$WT" || exit 1
mkdir -p perf/sizegate/campaign/logs perf/sizegate/campaign/claim-rr

# One runner per card, enforced by the kernel. The per-model mkdir claim below serialises
# MODELS, not CARDS, so two runners pointed at one card each take a DIFFERENT model and then
# fold both on that card at once -- which is what the claim looks like it prevents. Two
# things break at once when it happens: the runtimes are contention, not the model, and both
# passes share RELEASE_GATE_SIZE_WORKDIR, so whichever finishes first rmtree's the other's
# scratch out from under it. Hit on 2026-09-20 within four minutes of launching card 2 twice.
#
# flock, not a pid file: the lock is held by the fd and released when the process dies, so a
# killed runner leaves nothing stale behind to unblock by hand.
LOCK=perf/sizegate/campaign/card$CARD.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[rr card $CARD] another runner already holds card $CARD, refusing to share it"
  exit 4
fi

fresh() {
  "$PY" - "$1" <<'PYEOF'
import importlib.util, json, pathlib, sys
spec = importlib.util.spec_from_file_location("rg", "scripts/release_gate.py")
rg = importlib.util.module_from_spec(spec); spec.loader.exec_module(rg)
f = pathlib.Path("docs/size_ladder_baseline.d") / (sys.argv[1] + ".json")
try:
    e = json.loads(f.read_text())["cards"]["p150a"]["models"][sys.argv[1]]
except Exception:
    sys.exit(1)
sys.exit(0 if rg._size_ladder_same_engine(e.get("commit"), "HEAD") else 1)
PYEOF
}

healthy() { "$PY" perf/sizegate/campaign/card_health.py "$CARD"; }

revive() {
  echo "[rr card $CARD] ARC dead, resetting $(date -u +%FT%TZ)"
  timeout 300 "$HOME/.local/bin/tt-smi" -r "$CARD" 2>&1 | tail -2
  sleep 10
  healthy
}

for M in "$@"; do
  if fresh "$M"; then echo "[rr card $CARD] $M already recorded at this engine"; continue; fi
  if ! healthy && ! revive; then
    echo "[rr card $CARD] ARC still dead after reset, abandoning $(date -u +%FT%TZ)"; exit 3
  fi
  if ! mkdir "perf/sizegate/campaign/claim-rr/$M" 2>/dev/null; then
    echo "[rr card $CARD] $M claimed by another card"; continue
  fi
  LOG=perf/sizegate/campaign/logs/rr_$M.card$CARD.log
  echo "=== $M card $CARD start $(date -u +%FT%TZ) ===" >> "$LOG"
  PYTHONPATH="$WT" RELEASE_GATE_CENSUS_PYTHONPATH="$WT" \
  RELEASE_GATE_SIZE_WORKDIR="$WT/perf/sizegate/work-card$CARD" \
  TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=0,$CARD \
  TT_BIO_LEASE_HOLDER=worker:cov-ladder-p150a-p3 \
    "$PY" scripts/release_gate.py --model size-ladder --size-ladder-record \
      --size-ladder-fragment --size-ladder-models "$M" --load-ceiling 0 >> "$LOG" 2>&1
  echo "=== $M card $CARD rc=$? $(date -u +%FT%TZ) ===" >> "$LOG"
  if fresh "$M"; then
    echo "=== $M card $CARD fragment now carries this engine ===" >> "$LOG"
  else
    rmdir "perf/sizegate/campaign/claim-rr/$M" 2>/dev/null
    if ! healthy; then
      echo "=== $M card $CARD failed WITH A DEAD ARC: the card, not the model ===" >> "$LOG"
      revive || { echo "[rr card $CARD] unrecoverable $(date -u +%FT%TZ)"; exit 3; }
    fi
  fi
done
echo "[rr card $CARD] slice finished $(date -u +%FT%TZ)"

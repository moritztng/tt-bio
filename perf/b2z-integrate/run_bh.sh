#!/usr/bin/env bash
# b2z: run the integrated arm on qb2 Blackhole under benchlock, against its own incumbent.
#
# Usage, from inside the worktree that carries the arm:
#   perf/b2z-integrate/run_bh.sh <card> <out-tag> --arm 'name=mod.attr=1' [--arm ...] [--combined all]
#
# Card, venv and benchlock are the only host facts here and none of them are hardcoded to a path
# below $HOME, so this runs unchanged in any worker's worktree on qb2.
set -euo pipefail

CARD="${1:?card index, e.g. 1 -- must match the lease this row holds}"
TAG="${2:?output tag, e.g. arm1}"
shift 2

WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${TT_BIO_PY:-$HOME/tt-bio-dev/env/bin/python3}"
BENCHLOCK="${BENCHLOCK:-$HOME/.coworker/scripts/benchlock.sh}"
OUT="$WT/perf/b2z-integrate/${TAG}.json"
CIF="$WT/perf/b2z-integrate/cif_${TAG}"
LOG="$WT/perf/b2z-integrate/${TAG}.log"

[ -x "$PY" ] || { echo "no venv python at $PY" >&2; exit 2; }
[ -x "$BENCHLOCK" ] || { echo "no benchlock at $BENCHLOCK" >&2; exit 2; }

# An absolute number off a contended box is worthless, and the A/B ratio is only trustworthy if
# nothing else lands mid-run. benchlock holds the box for the whole session, not per fold
# (memory benchlock-one-shot-check-blind-to-mid-run-contention).
# NOT `exec`: this is a pipeline, so exec would only replace the left-hand subshell. `pipefail`
# above is what makes benchlock's exit 75 (lock timeout, do not measure anyway) reach the caller
# instead of being masked by tee's 0.
"$BENCHLOCK" b2z-integrate -- env \
  TT_VISIBLE_DEVICES="$CARD" \
  TT_BIO_LEASE_CARDS="$CARD" \
  TT_BIO_LEASE_HOLDER="worker:b2z-integrate" \
  PYTHONPATH="$WT" \
  "$PY" "$WT/perf/b2z-integrate/ab_arms.py" \
    --out "$OUT" --cifdir "$CIF" --reps 5 "$@" 2>&1 | tee "$LOG"

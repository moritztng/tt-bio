#!/bin/bash
# Finish the release gate: the one remaining arm (size-ladder), then compose the verdict.
#
# Why this arm is NOT split per model, though the box keeps resetting. The journal key includes
# `commit` and `dirty`, so any edit to scripts/release_gate.py in this worktree invalidates all
# TWELVE arms already banked at 135aeb3f0. Per-model resumability needs a code change
# (_arm_members returns ["size-ladder"] regardless of --size-ladder-models, so a one-model run
# would discharge the nine-model arm -- a real defect, fixed on wk/pvx-orchestrator, NOT here).
# Paying twelve arms to make the thirteenth cheaper is the wrong trade.
#
# Timing evidence for the retry: the 17:19Z run reached six of nine models in 1h43m before qb2
# hard-reset at 19:35:53. boltz2 8 min, esmfold2 19, protenix-v1 6, protenix-v2 20, openfold3 18,
# opendde 31, rf3 was 30 min in. Estimate ~2h45m for all nine.
set -u
WT=/home/ttuser/.coworker/wt/pvx-land
KEEP=/home/ttuser/pvx_gate_outputs_20260919T1621Z
LOG=/home/ttuser/pvx_arms/arms.log
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export PYTHONPATH="$WT"          # score the BRANCH tree, not the editable tt-bio-dev install
export TT_BIO_AICLK=1350
export TT_VISIBLE_DEVICES=3
export TT_BIO_LEASE_CARDS=3
export TT_BIO_LEASE_HOLDER=worker:pvx-orchestrator

say() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

tidy() {
  git status --porcelain | sed 's/^?? //' | while read -r p; do
    d="$KEEP/$(dirname "$p")"; mkdir -p "$d"; mv "$p" "$d/" 2>/dev/null || true
  done
  say "tidy: $(git status --porcelain -- tt_bio scripts tests pyproject.toml | wc -l) source paths dirty (0 = resumable)"
}

say "=== size-ladder retry after the 19:35:53Z hard reset, tree $(git rev-parse --short HEAD) ==="
tidy
say "--- arm size-ladder begins (attempt 2)"
BENCHLOCK_MAXLOAD=3.0 BENCHLOCK_WAIT_S=7200 \
  bash /home/ttuser/.coworker/scripts/benchlock.sh pvx-orchestrator -- \
  $PY scripts/release_gate.py --model size-ladder --keep >> "$LOG" 2>&1
rc=$?
say "--- arm size-ladder ends rc=$rc"

tidy
say "=== composing the full verdict from the journal (no arm should re-fold) ==="
$PY scripts/release_gate.py --resume >> "$LOG" 2>&1
say "=== compose ends rc=$? ==="

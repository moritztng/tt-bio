#!/bin/bash
# Re-gate the merge candidate 555d597b7 (the wk/pvx-orchestrator tip) after the previous gate's
# journal was destroyed on 2026-09-19 at 21:33Z.
#
# WHAT WAS LOST. The gate ran out of /home/ttuser/.coworker/wt/pvx-land with its journal at
# <worktree>/perf/gate_journal/journal.jsonl. The pvx-land row concluded at 19:03Z; fleet hygiene
# removed its worktree at 21:33Z while this gate was rooted there. Twelve banked PASS records at
# 135aeb3f0 went with it, the running job lost its cwd ("fatal: Unable to read current working
# directory"), and the compose died rc=2 on a scripts/ path that no longer existed. A journal
# written to survive a killed process does not survive a deleted directory.
#
# THE TWO FIXES, both structural rather than procedural:
#   1. the tree is /home/ttuser/pvx_gate_wt, OUTSIDE .coworker/wt, so no slug's conclusion can
#      take it. of3t-reopen-wt and aiclk-h already live this way and both survived the teardown.
#   2. the journal is /home/ttuser/pvx_arms/journal_555d597b7.jsonl, outside ANY worktree, via
#      release_gate.py --journal. A crash record has no business living inside the thing whose
#      disappearance it exists to survive.
#
# WHY 555d597b7 RATHER THAN 135aeb3f0. The old gated tree was one merge behind: main gained
# pvx-eligibility's five _MM_BLOCK fused-qkv keys at 18:02Z and they have never been gated.
# 555d597b7 is origin/main + those keys + TT_BIO_SDPA_WIDE_K default ON + the gate fixes, i.e.
# exactly the tree that would merge. Losing the journal removed the only reason to keep scoring
# the older tree, so the re-run buys a gate on the real merge candidate.
#
# WHY TWO BENCHLOCK ROUNDS. qb2 carries c14-land-tail holding the lock and pvx-baseline queued
# behind it. One round for the twelve correctness arms, released, then a second for size-ladder,
# so a timed co-tenant gets a turn in between instead of waiting out the whole gate.
set -u
WT=/home/ttuser/pvx_gate_wt
JOURNAL=/home/ttuser/pvx_arms/journal_555d597b7.jsonl
KEEP=/home/ttuser/pvx_gate_outputs_555d597b7
LOG=/home/ttuser/pvx_arms/gate555.log
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
cd "$WT" || exit 1
mkdir -p "$KEEP"
export PYTHONPATH="$WT"          # score THIS tree, not the editable tt-bio-dev install
export TT_BIO_AICLK=1350
export TT_VISIBLE_DEVICES=1      # card 2 is c14-land-tail's live lease; 1 is free
export TT_BIO_LEASE_CARDS=1
export TT_BIO_LEASE_HOLDER=worker:pvx-orchestrator

say() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

say "=== gate 555d597b7 starts, tree $(git rev-parse --short HEAD), journal $JOURNAL ==="
$PY -c "import tt_bio,sys; sys.stderr.write('resolved tt_bio: '+tt_bio.__file__+chr(10))" 2>&1 \
  | grep resolved >> "$LOG"

BENCHLOCK_MAXLOAD=4.0 BENCHLOCK_WAIT_S=21600 \
  bash "$BL" pvx-orchestrator -- bash /home/ttuser/pvx_arms/gate555_phase1.sh >> "$LOG" 2>&1
say "--- phase 1 ends rc=$?"

say "--- phase 2: size-ladder, its own benchlock round"
BENCHLOCK_MAXLOAD=3.0 BENCHLOCK_WAIT_S=21600 \
  bash "$BL" pvx-orchestrator -- \
  $PY scripts/release_gate.py --model size-ladder --journal "$JOURNAL" --keep >> "$LOG" 2>&1
say "--- arm size-ladder ends rc=$?"

say "=== composing the full verdict from the journal (no arm should re-fold) ==="
cd "$WT" && $PY scripts/release_gate.py --resume --journal "$JOURNAL" >> "$LOG" 2>&1
say "=== compose ends rc=$? ==="

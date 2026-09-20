#!/bin/bash
# Finish the release gate at the wk/pvx-land tree, one arm per process.
#
# Why per-arm rather than one driver. qb2 hard-reset at 16:21:00Z on 2026-09-19 and took a gate that
# had nine arms green behind it -- the third such kill in two days (gate_journal.py's own docstring
# records 11:03:23Z and 19:05:50Z on 09-18). One driver for thirteen arms means every reset costs
# every arm not yet composed into the final verdict. One process per arm banks each verdict in the
# journal as it decides, so a reset costs the arm in flight and nothing else. release_gate.py:1045
# describes this launcher as the intended use of --list-arms.
#
# Why tidy() runs before every arm. release_gate.py resumes on a key that includes `dirty`, and
# gate_journal.resumable() returns {} outright for a dirty tree. The gate writes its own run outputs
# (boltz2_results_prot/, pxdesign_gate.json, ...) into the repo ROOT, and _repo_dirty() greps
# `git status --porcelain` with no pathspec, so the gate dirties its own tree and every arm after the
# first journals as unresumable. That is why the nine arms already green were unresumable and why
# this run would otherwise have re-folded all of them. Moving the outputs aside (never deleting)
# before each arm holds the tree at its committed identity. _JOURNAL_KEY is computed once at
# release_gate.py:4774, so tidying before the process starts is enough.
#
# Why benchlock wraps exactly one arm. size-ladder is the only remaining arm that TIMES folds and
# checks a scaling exponent; the other four are correctness arms. Holding the box exclusively for
# all five would block every co-tenant on qb2 for hours to protect four arms that do not need it,
# and benchlock.sh's own header says to lock the measurement rather than the task.
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
  say "tidy: $(git status --porcelain | wc -l) dirty paths remain (0 = resumable)"
}

say "=== remaining-arms run starts, tree $(git rev-parse --short HEAD) ==="
$PY -c "import tt_bio,sys; sys.stderr.write('resolved tt_bio: '+tt_bio.__file__+chr(10))" 2>&1 \
  | grep resolved >> "$LOG"

for arm in l1-budget batch-position esmc-300m esmc-600m size-ladder; do
  tidy
  say "--- arm $arm begins"
  if [ "$arm" = size-ladder ]; then
    # the one timed arm: take the box before the clock starts
    BENCHLOCK_MAXLOAD=3.0 BENCHLOCK_WAIT_S=7200 \
      bash /home/ttuser/.coworker/scripts/benchlock.sh pvx-orchestrator -- \
      $PY scripts/release_gate.py --model "$arm" --keep >> "$LOG" 2>&1
  else
    $PY scripts/release_gate.py --model "$arm" --keep >> "$LOG" 2>&1
  fi
  say "--- arm $arm ends rc=$?"
done

tidy
say "=== composing the full verdict from the journal (no arm should re-fold) ==="
$PY scripts/release_gate.py --resume >> "$LOG" 2>&1
say "=== compose ends rc=$? ==="

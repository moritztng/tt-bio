#!/bin/bash
# Phase 1 of the 555d597b7 gate: the twelve correctness arms, one process per arm.
#
# One process per arm because qb2 hard-reset three times in two days and a single driver loses
# every arm it has not composed. Each process banks its verdict in the journal as it decides.
#
# Run me UNDER benchlock (the driver does that), not because a correctness arm needs a quiet box
# but because folding outside the lock while a timing row holds it is the instrument defect this
# campaign found in pvx-eligibility: its unlocked fold blocked the lock holder for 206 s.
set -u
WT=/home/ttuser/pvx_gate_wt
JOURNAL=/home/ttuser/pvx_arms/journal_555d597b7.jsonl
KEEP=/home/ttuser/pvx_gate_outputs_555d597b7
LOG=/home/ttuser/pvx_arms/gate555.log
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
mkdir -p "$KEEP"

say() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

# Hold the tree at its committed identity. release_gate.py writes its own run outputs into the
# repo root under --keep, and the resume key carries `dirty`. The scoped _repo_dirty() landed in
# 6ca42482e makes this belt-and-braces rather than load-bearing, which is why it never deletes:
# every moved path is an output somebody may want to read.
tidy() {
  git status --porcelain | sed 's/^?? //' | while read -r p; do
    d="$KEEP/$(dirname "$p")"; mkdir -p "$d"; mv "$p" "$d/" 2>/dev/null || true
  done
  say "tidy: $(git status --porcelain -- tt_bio scripts tests pyproject.toml | wc -l) source paths dirty (0 = resumable)"
}

# The nine fold models are ONE arm (fold-models) and must be one process: a run recorded with
# --model boltz2 alone journals members=["boltz2"], which is not a superset of the nine, so the
# next --resume refuses it and re-folds all nine. Enumerated to match MODELS in release_gate.py.
FOLD="--model boltz2 --model esmfold2 --model esmfold2-fast --model protenix-v2 --model protenix-v1 --model opendde --model openfold3 --model rf3 --model openbind"

ARMS=(
  "fold-models|$FOLD"
  "boltzgen|--model boltzgen"
  "rfd3|--model rfd3"
  "rfd3-fusion|--model rfd3-fusion"
  "opendde-abag|--model opendde-abag"
  "pxdesign|--model pxdesign"
  "nesso1|--model nesso1"
  "rf3-1024aa|--model rf3-1024aa"
  "capacity|--model capacity"
  "l1-budget|--model l1-budget"
  "batch-position|--model batch-position"
  "esmc|--model esmc-300m --model esmc-600m"
)

for entry in "${ARMS[@]}"; do
  name=${entry%%|*}; args=${entry#*|}
  tidy
  say "--- arm $name begins"
  # shellcheck disable=SC2086
  $PY scripts/release_gate.py $args --journal "$JOURNAL" --keep >> "$LOG" 2>&1
  say "--- arm $name ends rc=$?"
done
tidy
say "--- phase 1 complete"

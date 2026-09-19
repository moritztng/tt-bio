#!/usr/bin/env bash
# Compose the OF3T row branches into `wk/of3t` and verify the result against a control.
#
# The row branches move, so the numbers in COMPOSITION.md are a snapshot and this script is the
# thing that regenerates them. Run it rather than trusting the table.
#
# Three checks, and the second is the one that matters:
#   1. every row branch merges clean, and every row is an ANCESTOR of the result afterwards --
#      six clean merges does not mean six rows present, because a row can push mid-compose;
#   2. test collection on the composition against a DETACHED checkout of origin/main, on the SAME
#      interpreter, comparing the ERROR SETS -- an absolute error count says nothing, because a
#      host without ttnn/torch extras reports ~106 errors on plain main too;
#   3. the CPU-only instruments recompute, rather than their committed JSON being believed.
#
# It does NOT verify behaviour: on a host without ttnn no test executes. Say so when reporting.
set -euo pipefail

# A row is listed here from the moment it is dispatched, not from its first push, so a new row
# cannot be silently left out of the composition. Rows with no branch yet are skipped with a line
# saying so -- silence would be the bug.
ROWS="reference tape equivalence data perf memory confidence"
SLUG_TMP="${SLUG_TMP:-/tmp/of3t/of3t-orchestrator}"   # slug-scoped, never a shared /tmp name
PY="${PY:-/home/moritz/of3-upstream-venv/bin/python3}"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CO="$SLUG_TMP/compose"; BASE="$SLUG_TMP/basemain"

cd "$REPO"; git fetch -q origin
rm -rf "$CO" "$BASE"; mkdir -p "$SLUG_TMP"; git worktree prune
git branch -f wk/of3t origin/main >/dev/null 2>&1 || git branch wk/of3t origin/main
git worktree add -q "$CO" wk/of3t
git worktree add -q --detach "$BASE" origin/main

cd "$CO"
PRESENT=""
for r in $ROWS; do
  if git rev-parse --verify -q "origin/wk/of3t-$r" >/dev/null; then
    git merge --no-edit -q "origin/wk/of3t-$r" || { echo "CONFLICT merging of3t-$r:"; \
      git diff --name-only --diff-filter=U; exit 1; }
    PRESENT="$PRESENT $r"
  else
    echo "of3t-$r: dispatched but has not pushed a branch yet, skipped"
  fi
done
git merge --no-edit -q wk/of3t-orchestrator || true

# (1) ancestry, asserted AFTER the merges
for r in $PRESENT; do
  git merge-base --is-ancestor "origin/wk/of3t-$r" HEAD \
    || { echo "of3t-$r is NOT in the composition -- it advanced mid-compose; rerun"; exit 1; }
done
echo "ancestry:$(echo $PRESENT | wc -w) of $(echo $ROWS | wc -w) rows in $(git rev-parse --short HEAD), $(git rev-list --count origin/main..HEAD) ahead of main"

# (1b) the ownership claim: no two row branches may touch the same file
dup=$(for r in $PRESENT; do git diff --name-only "origin/main...origin/wk/of3t-$r"; done \
      | sort | uniq -d)
[ -z "$dup" ] && echo "ownership: disjoint, 0 files touched by more than one row" \
              || { echo "OWNERSHIP COLLISION:"; echo "$dup"; exit 1; }

# (2) collection against the control
# pytest exits nonzero whenever anything failed to collect, and on a host without the ttnn/torch
# extras that is ~106 files on plain main too. That is the expected state here, not an error, so
# the exit status is deliberately discarded and the ERROR SET is what gets compared.
col() { ( cd "$1" && timeout 900 "$PY" -m pytest tests/ --collect-only -q 2>&1 ) || true; }
col "$BASE" > "$SLUG_TMP/col_base.txt"; col "$CO" > "$SLUG_TMP/col_comp.txt"
grep -E '^ERROR' "$SLUG_TMP/col_base.txt" | sort > "$SLUG_TMP/err_base.txt" || true
grep -E '^ERROR' "$SLUG_TMP/col_comp.txt" | sort > "$SLUG_TMP/err_comp.txt" || true
echo "main:        $(tail -1 "$SLUG_TMP/col_base.txt")"
echo "composition: $(tail -1 "$SLUG_TMP/col_comp.txt")"
diff -q "$SLUG_TMP/err_base.txt" "$SLUG_TMP/err_comp.txt" >/dev/null \
  && echo "error sets: IDENTICAL -- no new import breakage" \
  || { echo "NEW IMPORT BREAKAGE:"; diff "$SLUG_TMP/err_base.txt" "$SLUG_TMP/err_comp.txt"; exit 1; }
echo "NOTE: no test EXECUTED -- without ttnn everything errors on extras or skips. Behaviour unverified."

# (2b) a reference to a file that does not exist. Composing found exactly this in NOTICE last
# pass, on a branch that looked fine alone, so it is a standing check rather than a one-off.
missing=""
for d in $(git diff origin/main...HEAD | grep -oE 'docs/[a-z0-9_-]+\.md' | sort -u); do
  [ -f "$d" ] || missing="$missing $d"
done
[ -z "$missing" ] && echo "doc refs: every docs/*.md the diff mentions exists" \
                  || { echo "DANGLING DOC REFERENCE:$missing"; exit 1; }

# (3) recompute the CPU-only instruments
for i in instrument_b_lr instrument_c_optim; do
  f="perf/of3t_equivalence/$i.py"
  [ -f "$f" ] && { echo "--- $i"; ( cd "$CO" && PYTHONPATH=. timeout 1800 "$PY" "$f" 2>&1 | tail -8 ); }
done

git worktree remove --force "$BASE"
echo; echo "composition ready at $CO ; push with: git -C $CO push origin wk/of3t"

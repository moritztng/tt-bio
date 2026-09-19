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
ROWS="reference tape equivalence data perf memory confidence leaves gradients pairbias l1 updaterule"
SLUG_TMP="${SLUG_TMP:-/tmp/of3t/of3t-orchestrator}"   # slug-scoped, never a shared /tmp name
PY="${PY:-/home/moritz/of3-upstream-venv/bin/python3}"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CO="$SLUG_TMP/compose"; BASE="$SLUG_TMP/basemain"
D_WT="${D_WT:-/home/moritz/.coworker/wt}"   # worktrees on THIS host, for the unpushed-work note

cd "$REPO"
# A bare `git fetch origin` fails outright if ANOTHER campaign's worker is updating its refs at
# the same moment ("cannot lock ref refs/remotes/origin/wk/pvx-...") and takes the whole compose
# down for a reason that has nothing to do with this campaign. Fetch only what we read, and
# retry once, so a neighbour's push cannot abort the composition.
_refs="main"; for _r in $ROWS; do _refs="$_refs wk/of3t-$_r"; done
_refs="$_refs wk/of3t-orchestrator"
# shellcheck disable=SC2086
git fetch -q origin $_refs 2>/dev/null || { sleep 5; git fetch -q origin $_refs 2>/dev/null || {
  echo "fetch failed twice for this campaign's own refs -- not a neighbour race"; exit 1; }; }
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

# (1b) the ownership claim: no two rows may EDIT the same file.
#
# The obvious form of this check is wrong and was wrong here for a pass. Diffing
# `origin/main...origin/wk/of3t-$r` reports every file the row INHERITED as well as every file it
# edited, and rows are told to build on `wk/of3t` rather than on main -- so the moment one did,
# the check reported a 100-file "collision" in which the newest row claimed every other row's
# artifacts. It was measuring inheritance.
#
# A row's own files are the files touched by commits reachable from ITS branch and from no other
# row's branch. That is what `git log A --not B C D` gives.
others_of(){ local me="$1" r; for r in $PRESENT; do [ "$r" = "$me" ] || \
  printf 'origin/wk/of3t-%s ' "$r"; done; }
: > "$SLUG_TMP/own.txt"
for r in $PRESENT; do
  # shellcheck disable=SC2046
  for c in $(git rev-list "origin/wk/of3t-$r" --not origin/main $(others_of "$r")); do
    git diff-tree --no-commit-id --name-only -r "$c"
  done | sort -u | sed "s|^|$r |" >> "$SLUG_TMP/own.txt"
done
# Declared co-edits: a file two rows were BOTH granted, deliberately, in disjoint regions.
# Blanket-failing on these would abort every compose and teach me to ignore the check, which is
# how a gate dies. Each entry needs a reason and the regions have to be verified disjoint when it
# is added -- an entry here is a claim, not a mute button.
#   tt_bio/tenstorrent.py: of3t-leaves owns the weight-discovery seam on `Module` (~5857-5890);
#   of3t-confidence owns the confidence path's `PairformerLayer`/`Pairformer` plumbing
#   (~8764-8953). ~2900 lines apart, different classes, verified 2026-09-19 pass 6.
ALLOWED_COEDIT="tt_bio/tenstorrent.py"

dup=$(awk '{print $2}' "$SLUG_TMP/own.txt" | sort | uniq -d)
for a in $ALLOWED_COEDIT; do
  if printf '%s\n' "$dup" | grep -qx "$a"; then
    printf 'ownership: %s co-edited by' "$a"
    grep " $a\$" "$SLUG_TMP/own.txt" | awk '{printf " %s",$1}'
    echo " -- DECLARED, regions verified disjoint"
  fi
  dup=$(printf '%s\n' "$dup" | grep -vx "$a" || true)
done
dup=$(printf '%s\n' "$dup" | sed '/^$/d')
if [ -z "$dup" ]; then
  echo "ownership: no undeclared file is edited by more than one row"
else
  echo "OWNERSHIP COLLISION -- these files are edited by more than one row:"
  while read -r f; do printf '  %s  <-' "$f"; grep " $f\$" "$SLUG_TMP/own.txt" \
    | awk '{printf " %s",$1}'; echo; done <<< "$dup"
  exit 1
fi

# (1c) UNPUSHED WORK. A composition can only ever see `origin`, and a row whose worktree is ahead
# of its remote is invisible to it -- not stalled, invisible. `of3t-reference` sat at its
# pass-1 commit on origin for five passes while its worktree held two commits, an untracked
# generator and twenty frozen batches, and the campaign's critical path was diagnosed from
# `origin` twice and called stuck. This is a WARNING rather than a failure: the row owns its
# branch and may be mid-work, and pushing for it would race a live agent.
for r in $ROWS; do
  wt="$D_WT/of3t-$r"
  [ -d "$wt" ] || continue
  ah=$(git -C "$wt" rev-list --count "origin/wk/of3t-$r..HEAD" 2>/dev/null || echo 0)
  dirty=$(git -C "$wt" status --porcelain 2>/dev/null | wc -l)
  [ "$ah" -gt 0 ] && echo "  NOTE of3t-$r: worktree is $ah commit(s) ahead of origin -- not in this composition"
  [ "$dirty" -gt 0 ] && echo "  NOTE of3t-$r: $dirty uncommitted file(s) in its worktree"
  # WHY it is behind. `rc=124` is the 3000s per-turn cap: the row commits and is killed before
  # it pushes, which looks identical to disobedience from `origin` and is not. Asking such a
  # row to push cannot work -- its work has to be reshaped to fit the wall. (K38; this cost
  # two passes of wrong remedies on of3t-reference.)
  lg="$D_WT/../workers/of3t-$r.log"
  if [ -f "$lg" ] && [ "$ah" -gt 0 ]; then
    last=$(grep -oE 'it[0-9]+ rc=[0-9]+' "$lg" | tail -1)
    case "$last" in *rc=124) echo "  NOTE of3t-$r: last turn was KILLED by the 3000s cap ($last)"\
      " -- it likely commits and never reaches a push; reshape the work, do not re-ask";; esac
  fi
done

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
for f in perf/of3t_equivalence/instrument_b_lr.py \
         perf/of3t_equivalence/instrument_c_optim.py \
         perf/of3t_orchestrator/instrument_b2_clip.py; do
  [ -f "$CO/$f" ] || continue
  echo "--- $(basename "$f" .py)"
  ( cd "$CO" && PYTHONPATH=. timeout 1800 "$PY" "$f" 2>&1 | tail -8 ) || \
    { echo "INSTRUMENT FAILED: $f"; exit 1; }
done

# (4) the scoreboard against the artifacts. EVIDENCE.md is transcribed prose and a
# transcription drifts silently, so the numbers it quotes are re-read from the committed JSON
# on every compose. Also pins the denominators (K29).
echo "--- audit_evidence"
( cd "$CO" && "$PY" perf/of3t_orchestrator/audit_evidence.py 2>&1 | tail -6 ) || \
  { echo "SCOREBOARD DRIFT -- state/of3t/EVIDENCE.md disagrees with the artifacts"; exit 1; }

git worktree remove --force "$BASE"
echo; echo "composition ready at $CO ; push with: git -C $CO push origin wk/of3t"

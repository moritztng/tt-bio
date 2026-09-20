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
ROWS="reference tape equivalence data perf memory confidence leaves gradients pairbias l1 updaterule entity diffusion reopen rebase confhead auxheads"
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
# A row can be dispatched (and so listed in ROWS) before it has pushed a branch. The merge loop
# below tolerates that, but `git fetch` does NOT: naming one absent ref fails the whole fetch,
# and the retry message then blames a neighbour race for a row that simply has not started.
# Ask origin what exists first, and fetch only that.
_avail="$(git ls-remote --heads origin 2>/dev/null | sed 's#.*refs/heads/##')"
[ -n "$_avail" ] || { sleep 5; _avail="$(git ls-remote --heads origin 2>/dev/null | sed 's#.*refs/heads/##')"; }
[ -n "$_avail" ] || { echo "cannot list origin refs -- network or remote is down"; exit 1; }
_refs="main"
for _r in $ROWS; do
  printf '%s\n' "$_avail" | grep -qx "wk/of3t-$_r" && _refs="$_refs wk/of3t-$_r"
done
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
    if ! git merge --no-edit -q "origin/wk/of3t-$r"; then
      # `.gitignore` is the one file where a conflict is routinely NOT a disagreement: both
      # sides append an ignore rule for their own scratch, and keeping both is what each side
      # meant. Resolved by union, announced, and ONLY for this path -- any other conflicted
      # file still stops the composition, because for real content a union is a guess.
      _u=$(git diff --name-only --diff-filter=U)
      if [ "$_u" = ".gitignore" ]; then
        git checkout --theirs .gitignore 2>/dev/null || true
        git show :2:.gitignore > /tmp/.gi_ours 2>/dev/null
        git show :3:.gitignore > /tmp/.gi_theirs 2>/dev/null
        cat /tmp/.gi_ours /tmp/.gi_theirs | awk '!seen[$0]++ || $0==""' > .gitignore
        rm -f /tmp/.gi_ours /tmp/.gi_theirs
        git add .gitignore && git commit --no-edit -q
        echo "  NOTE of3t-$r: .gitignore conflict resolved by UNION (both sides append their"\
             " own scratch rule); every other path would have stopped the compose"
      else
        echo "CONFLICT merging of3t-$r:"; printf '%s\n' "$_u"; exit 1
      fi
    fi
    PRESENT="$PRESENT $r"
  else
    echo "of3t-$r: dispatched but has not pushed a branch yet, skipped"
  fi
done
# NOT `|| true`. This merge lands last and is therefore the only thing that can override a row,
# which is exactly why its failure must be loud: a conflict here silently drops the orchestrator's
# corrections from the branch that goes to the merge gate, and the compose would still print
# "clean". Found pass 92 while reverting a default flip through this very merge.
git merge --no-edit -q wk/of3t-orchestrator \
  || { echo "CONFLICT merging wk/of3t-orchestrator:"; git diff --name-only --diff-filter=U; exit 1; }

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
#   tt_bio/train/optim.py: of3t-updaterule owns `AdamW.step` (~192-206, where the schedule is
#   read relative to the counter, D11); of3t-gradients owns `displacement` and
#   `check_displacement` (~289-336, the D13 resolution floor). Different methods, ~85 lines
#   apart, hunk ranges compared 2026-09-19 pass 37. NOTE the semantic coupling, which
#   disjointness does NOT cover: D13's relaxation was justified by a 0.810 displacement ratio
#   measured under the pre-D11 schedule read, and after D11 the same arm moves strictly less.
#   of3t-updaterule's brief is amended to re-state that number under the merged code.
ALLOWED_COEDIT="tt_bio/tenstorrent.py tt_bio/train/optim.py"

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
THIS_HOST="$(hostname -s)"
for r in $ROWS; do
  wt="$D_WT/of3t-$r"
  lg="$D_WT/../workers/of3t-$r.log"
  # A row whose worktree is on ANOTHER host has no directory here, and `continue` used to skip
  # it in silence -- so for every carded row this check has been reporting nothing while
  # looking like it reported "clean". That is the failure mode this check exists to catch,
  # turned on itself. When qb2 hard-hung on 2026-09-19 the two rows holding the campaign's
  # critical path were both there, and the compose said nothing about either.
  if [ ! -d "$wt" ]; then
    # `|| true` is load-bearing under `set -o pipefail`: for a row that is dispatched but has
    # never launched on ANY host there is no log file, `grep` exits non-zero, the pipeline
    # inherits it and the whole compose aborts with rc=2 on a completely normal state. Caught
    # pass 90 the first time a row was listed in ROWS before its first launch -- which is
    # exactly the case the ROWS comment says must be supported.
    rhost=$( { grep -oE 'START of3t-[a-z0-9]+ host=[^ ]+' "$lg" 2>/dev/null || true; } | \
            tail -1 | sed 's/.*host=//')
    # Only for a LIVE row. A concluded row's worktree holds nothing the campaign is waiting
    # on -- what origin has IS its final answer -- and noting nine of them buries the one
    # that matters, which is how a check gets ignored.
    if [ -n "$rhost" ] && [ "$rhost" != "$THIS_HOST" ] \
       && [ ! -f "$D_WT/../state/concluded/of3t-$r" ]; then
      echo "  NOTE of3t-$r: worktree is on $rhost, not $THIS_HOST -- unpushed work there is"\
           " INVISIBLE to this compose. What origin has is all this composition can carry."
    fi
    continue
  fi
  ah=$(git -C "$wt" rev-list --count "origin/wk/of3t-$r..HEAD" 2>/dev/null || echo 0)
  dirty=$(git -C "$wt" status --porcelain 2>/dev/null | wc -l)
  [ "$ah" -gt 0 ] && echo "  NOTE of3t-$r: worktree is $ah commit(s) ahead of origin -- not in this composition"
  [ "$dirty" -gt 0 ] && echo "  NOTE of3t-$r: $dirty uncommitted file(s) in its worktree"
  # WHY it is behind. `rc=124` is the 3000s per-turn cap: the row commits and is killed before
  # it pushes, which looks identical to disobedience from `origin` and is not. Asking such a
  # row to push cannot work -- its work has to be reshaped to fit the wall. (K38; this cost
  # two passes of wrong remedies on of3t-reference.)
  if [ -f "$lg" ] && [ "$ah" -gt 0 ]; then
    # `|| true` again, and this is the THIRD site in this script where `set -o pipefail` plus a
    # grep whose empty result is the NORMAL case took the whole compose down (the other two are
    # the remote-host probe above and the ownership table below). A row that is ahead of origin
    # but whose log has no `itN rc=N` line yet -- a first pass still running -- is completely
    # ordinary, and it aborted the compose at rc=1 right after printing a healthy ancestry line.
    # Pattern, named so the next grep added here gets it right: in this script every
    # `x=$(grep ... | ...)` needs `|| true`, because none of them treat "no match" as an error.
    last=$(grep -oE 'it[0-9]+ rc=[0-9]+' "$lg" 2>/dev/null | tail -1 || true)
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
  || { echo "NEW COLLECTION ERRORS (classified below):"
       # `|| true`: diff exits 1 when the files differ, which is the ONLY case this branch
       # runs in, and `set -e` then kills the script before the classifier can say a word.
       # Second time this campaign has been bitten by a command whose failure IS the expected
       # case (K65: `grep -v` filtering everything out under pipefail).
       diff "$SLUG_TMP/err_base.txt" "$SLUG_TMP/err_comp.txt" || true
       # Say WHAT KIND of new error it is. 84 of the ~107 baseline errors on a CPU host are a
       # top-level `import ttnn`, and three tests already avoid that with
       # `pytest.importorskip("ttnn")` and skip instead. A new file of that class is a
       # one-line convention miss, not broken code -- and a reader who has to run pytest by
       # hand to learn which it is will start ignoring this gate.
       _ttnn_only=0; _real=0
       for _f in $( { diff "$SLUG_TMP/err_base.txt" "$SLUG_TMP/err_comp.txt" || true; } \
                   | grep '^> ERROR' | awk '{print $3}'); do
         # Capture THEN grep. `pytest | grep -q` under `set -o pipefail` is non-zero whenever
         # pytest is -- which is always here, since the file is in the error set -- so the
         # pipeline reported "not the ttnn class" for a file that plainly was one. Third time
         # this script has been bitten by pipefail on a command whose failure is expected.
         _out=$( { cd "$CO" && "$PY" -m pytest "$_f" --collect-only -q 2>&1; } || true )
         if printf '%s' "$_out" | grep -q "No module named 'ttnn'"; then
           echo "  $_f: top-level \`import ttnn\` on a host without it -- the class 84 of the"
           echo "    baseline errors already are. The convention that avoids it is"
           echo "    \`pytest.importorskip(\"ttnn\")\` (see tests/test_sdpa_cb_model.py,"
           echo "    tests/test_training_full_weights.py, tests/test_sdpa_fused_pairs.py),"
           echo "    which SKIPS instead. One line, and the error set stays identical."
           _ttnn_only=$((_ttnn_only + 1))
         else
           echo "  $_f: NOT the ttnn class -- read it, this one may be real."
           _real=$((_real + 1))
         fi
       done
       # FAIL only on a new error that is NOT the ttnn class. A device test with a top-level
       # `import ttnn` is indistinguishable on pc from the 84 baseline errors of that shape,
       # and on a host WITH ttnn it collects fine -- so counting it as breakage measures "a
       # row added a device test", not "the composition broke imports". Keeping it fatal would
       # block every live row's normal work on a style preference, which is how a correctness
       # gate gets routed around. It still WARNS by name with the one-line convention fix.
       if [ "${_real:-0}" -gt 0 ]; then
         echo "  -> $_real new error(s) of an unknown class: that is breakage. Stopping."
         exit 1
       fi
       echo "  -> all $_ttnn_only new error(s) are the ttnn class; not breakage, but owed a"
       echo "     \`pytest.importorskip\` so the error set goes back to identical."; }
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
         perf/of3t_orchestrator/instrument_b2_clip.py \
         perf/of3t_orchestrator/instrument_c2_clip_in_step.py; do
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

# (5c) WILL IT MERGE? The composition is built by merging rows into a branch based on
# `origin/main`, which makes it *likely* to merge back cleanly and proves nothing. Main moves
# under us -- a `.gitignore` conflict already stopped one compose -- and the gate's question is
# not "did the rows compose" but "will this land". A trial merge in a throwaway worktree costs
# seconds and cannot be wrong about what git will do; every other way of answering can.
# `rm -rf` leaves git's worktree metadata behind, so a second run's `add` fails on a path that
# looks absent -- and the first version of this check swallowed that failure and printed
# NOTHING, which is the silently-skipped shape this compose exists to catch. Prune first, and
# say so if the add still fails.
rm -rf "$SLUG_TMP/mergetest"; git worktree prune
if ! git worktree add -q --detach "$SLUG_TMP/mergetest" origin/main 2>/dev/null; then
  echo "  NOTE merge gate did NOT run: could not create the trial worktree. Unproven, not clean."
else
  if git -C "$SLUG_TMP/mergetest" merge --no-commit --no-ff "$(git -C "$CO" rev-parse HEAD)" \
       >/dev/null 2>&1; then
    if git merge-base --is-ancestor origin/main "$(git -C "$CO" rev-parse HEAD)"; then
      echo "merge gate: clean, and the composition is a DESCENDANT of origin/main (fast-forwardable)"
    else
      echo "merge gate: clean (a real merge, not a fast-forward)"
    fi
  else
    echo "MERGE GATE: wk/of3t does NOT merge cleanly into origin/main. Conflicted:"
    git -C "$SLUG_TMP/mergetest" diff --name-only --diff-filter=U | sed 's/^/  /'
    git -C "$SLUG_TMP/mergetest" merge --abort 2>/dev/null || true
    git worktree remove --force "$SLUG_TMP/mergetest" 2>/dev/null || true
    exit 1
  fi
  git -C "$SLUG_TMP/mergetest" merge --abort 2>/dev/null || true
  git worktree remove --force "$SLUG_TMP/mergetest" 2>/dev/null || true
fi

# (5d) PUBLISH THE WRITTEN RECORD INTO THE BRANCH. The campaign's artifacts are in git; its
# REASONING is not -- PROTOCOL.md, DEFECTS.md, EVIDENCE.md and LEDGER.md live in
# `~/.coworker/state/of3t/`, which is gitignored, on one machine, with no backup. 440 KB of
# markdown holding every bar, every defect and every correction, one disk away from gone,
# while the 190 MB of artifacts they explain are replicated on origin. K61 said /tmp is scoped
# but not durable; `state/` is durable but not REPLICATED, which is the same lesson one level
# up. A reviewer handed `wk/of3t` should get the evidence AND the argument.
#
# Copies, not moves: `state/of3t/` stays authoritative and each copy says so in its header, so
# nobody edits the wrong one and the campaign never holds two live answers to one question.
_REC="$REPO/perf/of3t_orchestrator/record"
mkdir -p "$_REC"
_recn=0
# ORCHESTRATOR.md is the state doc, and it is here for the same reason as the rest: its
# PROVES / DOESNOT / VERDICT fields are the campaign's headline and the audit checks them, so
# a reviewer who cannot read them cannot run the checks that read them.
for _f in PROTOCOL DEFECTS EVIDENCE LEDGER ORCHESTRATOR; do
  _src="/home/moritz/.coworker/state/of3t/$_f.md"
  [ "$_f" = ORCHESTRATOR ] && _src="/home/moritz/.coworker/state/of3t-orchestrator.md"
  [ -f "$_src" ] || continue
  { echo "<!-- PUBLISHED COPY, regenerated by compose_verify.sh on every compose."
    echo "     AUTHORITATIVE SOURCE: ~/.coworker/state/of3t/$_f.md on pc."
    echo "     Edit that one. An edit here is overwritten on the next compose, and two live"
    echo "     copies of one document is the defect this campaign spent four passes fixing."
    echo "     Published so that wk/of3t carries the reasoning and not only the artifacts:"
    echo "     the source is gitignored, on one machine, with no backup. -->"
    echo
    cat "$_src"
  } > "$_REC/$_f.md"
  _recn=$((_recn + 1))
done
echo "record: $_recn campaign documents published into the branch ($(du -sh "$_REC" | cut -f1))"

# (6) REGENERATE COMPOSITION.md's data. Everything above is computed; the table in that file
# was TYPED, once, at pass 6 -- and by pass 57 it still said "the six row branches" and listed
# `of3t-confidence` as "dispatched, no branch yet", on the file a reviewer of `wk/of3t` opens
# first. Same drift class this campaign audits everywhere else, in a document I own. The prose
# stays authored; the numbers are written here, into $REPO so the branch carries them.
_DOC="$REPO/perf/of3t_orchestrator/COMPOSITION.md"
if [ -f "$_DOC" ] && grep -q '<!-- BEGIN GENERATED' "$_DOC"; then
  {
    echo "<!-- BEGIN GENERATED by compose_verify.sh -- do not edit between these markers -->"
    echo
    echo "Composed $(date -u +%F\ %TZ) from \`origin/main\` at \`$(git rev-parse --short origin/main)\`."
    echo "Head \`$(git -C "$CO" rev-parse --short HEAD)\`, **$(git -C "$CO" rev-list --count origin/main..HEAD) commits ahead**,"
    echo "carrying **$(set -- $PRESENT; echo $#) of $(set -- $ROWS; echo $#) rows**."
    echo
    echo "| row | branch head | files reachable ONLY from this row |"
    echo "|---|---|---|"
    for r in $ROWS; do
      if git rev-parse --verify -q "origin/wk/of3t-$r" >/dev/null; then
        _h=$(git rev-parse --short "origin/wk/of3t-$r")
        # `|| true`: under `set -o pipefail` a `grep -v` that filters everything out exits 1
        # and takes the whole compose down -- which it did, silently, for a row whose only
        # files are in its own namespace. The empty result is the NORMAL case here.
        _f=$(grep "^$r " "$SLUG_TMP/own.txt" | awk '{print $2}' | grep -v "^perf/of3t_$r/" \
             | sed 's/^/`/;s/$/`/' | paste -sd', ' - || true)
        [ -n "$_f" ] || _f="none outside its own \`perf/of3t_$r/\`"
        echo "| \`of3t-$r\` | \`$_h\` | $_f |"
      else
        echo "| \`of3t-$r\` | *dispatched, no branch yet* | — |"
      fi
    done
    echo
    echo "**Shape of the diff**, because 300-odd files is three very different piles:"
    echo
    echo "| pile | files | lines | what it is |"
    echo "|---|---|---|---|"
    _eng=$(git -C "$CO" diff --name-only origin/main HEAD | grep "^tt_bio/" | grep -v "_vendor" || true)
    if [ -n "$_eng" ]; then
      # shellcheck disable=SC2086
      _engstat=$(git -C "$CO" diff --shortstat origin/main HEAD -- $_eng | tr -d '\n')
      echo "| **engine** | $(printf '%s\n' "$_eng" | wc -l) under \`tt_bio/\`, none vendored | **$(echo "$_engstat" | grep -oE '[0-9]+ insertion[^,]*|[0-9]+ deletion[^,]*' | paste -sd' / ' -)** | the part that changes behaviour |"
    fi
    _ven=$(git -C "$CO" diff --shortstat origin/main HEAD -- tt_bio/_vendor/ | tr -d '\n')
    echo "| vendored upstream | $(git -C "$CO" diff --name-only origin/main HEAD -- tt_bio/_vendor/ | wc -l) under \`tt_bio/_vendor/openfold3/\` | $(echo "$_ven" | grep -oE '[0-9]+ insertion[^,]*|[0-9]+ deletion[^,]*' | paste -sd' / ' -) | their pipeline, carried in |"
    echo "| campaign artifacts | $(git -C "$CO" diff --name-only origin/main HEAD -- 'perf/of3t_*' | wc -l) under \`perf/of3t_*/\` | the bulk | measurements and instruments |"
    echo
    echo "Total $(git -C "$CO" diff --shortstat origin/main HEAD | tr -d '\n')."
    echo
    echo "The third column of the row table is what the **collision check** uses: files touched by commits on"
    echo "that branch and on no other row's. An **empty cell does not mean the row edited"
    echo "nothing outside its namespace** -- rows build on \`wk/of3t\`, so an early row's"
    echo "commits are reachable from every later row and drop out of its own unique set."
    echo "\`of3t-tape\` edited eight shared files and reads empty here. For collision"
    echo "detection that is the correct semantics (two rows uniquely touching one file is the"
    echo "defect); for \"who wrote this file\" it is not, and the git log is."
    echo
    echo "<!-- END GENERATED -->"
  } > "$SLUG_TMP/gen.md"
  "$PY" - "$_DOC" "$SLUG_TMP/gen.md" <<'PYGEN'
import re, sys, pathlib
doc, gen = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).read_text()
t = doc.read_text()
new = re.sub(r"<!-- BEGIN GENERATED.*?<!-- END GENERATED -->", gen.strip(), t, flags=re.S)
if new != t:
    doc.write_text(new); print("  COMPOSITION.md regenerated (commit it)")
else:
    print("  COMPOSITION.md already current")
PYGEN
fi

git worktree remove --force "$BASE"
# (6) SHIPPED DEFAULTS THE COMPOSITION MUST NOT MOVE.
#
# `wk/of3t` is what goes to Moritz's merge gate, so a default that changed inside it is a change
# that ships. `of3t-pairbias` measured the OF3 trunk pair-bias correction end to end and concluded
# "Land the MECHANISM. Do NOT flip the OF3 trunk default on this evidence" -- 0.050 A of best-of-5
# bought for 0.463 A on the structure a user receives, against a 0.324 A seed floor -- and then its
# own commit landed the flip anyway. It sat in the composition for passes; `of3t-confhead` found it
# by reading the branch rather than the verdict. The lever is one token, so the guard is one grep.
#
# This is deliberately NOT a general "no default moved" check, which would need a definition of
# `default` this script cannot honestly give. It is a named assertion about a named line, and when
# D1+D10 are approved to ship together the line here changes with them.
_trunk="$CO/tt_bio/openfold3_trunk.py"
_want='scale_pair_bias=False, tri_att_scale_pair_bias=False'
if grep -q "$_want" "$_trunk"; then
  echo "shipped defaults: OF3 trunk pair-bias default is False, matching main (D1 HELD: measured 0.170 A worse at rank 0, of3t-confhead pass 101; not blocked on D10, which is resolved)"
else
  echo "SHIPPED DEFAULT MOVED -- $_trunk does not carry: $_want"
  grep -n 'scale_pair_bias=' "$_trunk" | sed 's/^/  /'
  echo "  D1 is HELD on its own measurement: D1+D10 serves 0.170 A worse than shipped at rank 0 over"
  echo "  six seeds (of3t-confhead pass 101), and the best rule in the candidate set still serves"
  echo "  0.086 A worse. Fixing the selector did not rescue it. If Moritz approves it anyway,"
  echo "  change _want in this script in the same commit that flips the default."
  exit 1
fi

echo; echo "composition ready at $CO ; push with: git -C $CO push origin wk/of3t"
# `--push` exists because I once ran `compose_verify.sh; git -C $CO push -f` as one line and
# force-pushed a FAILED, half-merged composition over a good one: 256 commits replaced by 55.
# The branch is regenerated every pass so nothing was lost, but the shape of the mistake is
# permanent -- a push that is not conditional on the verdict is not a verified push.
if [ "${1:-}" = "--push" ]; then
  git -C "$CO" push -f -q origin wk/of3t \
    && echo "pushed wk/of3t -> $(git -C "$CO" rev-parse --short HEAD), \
$(git -C "$CO" rev-list --count origin/main..HEAD) ahead"
fi

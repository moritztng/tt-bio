#!/bin/bash
# The composition compose_verify.sh would have built, seeded at the PUBLISHED composition
# instead of at current origin/main.
#
# compose_verify.sh re-bases the composition onto origin/main every run. origin/main moved to
# fd70adde2 (the D56 softmax-backward merge) after the last publish, and `of3t-leaves` conflicts
# with it in tt_bio/autograd.py and tt_bio/tenstorrent.py. That stops the compose at row 8 of the
# floor, which is 30-odd rows before `barresolve` is ever merged -- so it says nothing about this
# row, and this row cannot fix it (autograd.py is shipped code in another row's hands).
#
# `origin/wk/of3t-leaves` merges CLEAN into `origin/wk/of3t`, so seeding there gets a tree that
# holds every row branch, this one included. Same merge loop, same row derivation from the
# briefs' #DISPATCH lines, same .gitignore union resolution, and every row that does NOT land is
# named rather than skipped quietly.
set -u
WS="${WS:-/tmp/of3t/of3t-barresolve/ws}"
REPO="${REPO:-/tmp/of3t/of3t-barresolve/repo}"
CO="${CO:-/tmp/of3t/of3t-barresolve/pubcompose}"
PY="${PY:-/home/ttuser/tt-bio-dev/env/bin/python}"

ROWS_FLOOR="reference tape equivalence data perf memory confidence leaves gradients pairbias l1 updaterule entity diffusion reopen rebase confhead auxheads conditioning adaln softmax trajectory"
ROWS_SEEN="$( { for _b in "$WS"/of3t-*.txt; do
    [ -f "$_b" ] || continue
    grep -q '^#DISPATCH:' "$_b" || continue
    _n="$(basename "$_b" .txt)"; _n="${_n#of3t-}"
    [ "$_n" = "orchestrator" ] && continue
    printf '%s\n' "$_n"
  done; } | sort -u | paste -sd' ' - )"
ROWS="$ROWS_FLOOR"
for _r in $ROWS_SEEN; do
  case " $ROWS_FLOOR " in *" $_r "*) ;; *) ROWS="$ROWS $_r" ;; esac
done

cd "$REPO" || exit 1
git worktree remove --force "$CO" 2>/dev/null; rm -rf "$CO"; git worktree prune
git branch -f wk/of3t-pub origin/wk/of3t >/dev/null 2>&1 || git branch wk/of3t-pub origin/wk/of3t
git worktree add -q "$CO" wk/of3t-pub || exit 1
cd "$CO" || exit 1

PRESENT=""; ABSENT=""; SKIPPED=""
for r in $ROWS; do
  git rev-parse --verify -q "origin/wk/of3t-$r" >/dev/null || { ABSENT="$ABSENT $r"; continue; }
  if ! git merge --no-edit -q "origin/wk/of3t-$r" >/dev/null 2>&1; then
    _u=$(git diff --name-only --diff-filter=U)
    if [ "$_u" = ".gitignore" ]; then
      git show :2:.gitignore > /tmp/.gi_o 2>/dev/null; git show :3:.gitignore > /tmp/.gi_t 2>/dev/null
      cat /tmp/.gi_o /tmp/.gi_t | awk '!seen[$0]++ || $0==""' > .gitignore; rm -f /tmp/.gi_o /tmp/.gi_t
      git add .gitignore && git commit --no-edit -q
      echo "  NOTE of3t-$r: .gitignore conflict resolved by UNION"
    else
      git merge --abort 2>/dev/null
      SKIPPED="$SKIPPED $r($(echo $_u | tr '\n' ',' | cut -c1-60))"
      continue
    fi
  fi
  PRESENT="$PRESENT $r"
done

echo "== rows present:$(echo $PRESENT | wc -w)  absent(no branch):$(echo $ABSENT | wc -w)  skipped(conflict):$(echo $SKIPPED | wc -w)"
[ -n "$ABSENT" ]  && echo "   no branch yet:$ABSENT"
[ -n "$SKIPPED" ] && echo "   CONFLICTED, not in this tree:$SKIPPED"

# Six clean merges is not six rows present -- assert ancestry, the way compose_verify does.
_bad=""
for r in $PRESENT; do
  git merge-base --is-ancestor "origin/wk/of3t-$r" HEAD || _bad="$_bad $r"
done
[ -n "$_bad" ] && { echo "NOT ANCESTORS:$_bad"; exit 1; }
echo "ancestry: all $(echo $PRESENT | wc -w) merged rows are ancestors of $(git rev-parse --short HEAD)"
echo "barresolve in tree: $(git merge-base --is-ancestor origin/wk/of3t-barresolve HEAD && echo YES || echo NO)"
echo "trajbar.py sha: $(git hash-object perf/of3t_trajbar/trajbar.py)  (branch: $(git --git-dir=$REPO/.git rev-parse origin/wk/of3t-barresolve:perf/of3t_trajbar/trajbar.py))"
echo
echo "--- sys.path order: no new tree-resolution trust (D149), the composition's own copy"
"$PY" perf/of3t_orchestrator/assert_path_order_ratchet.py .
echo "ratchet rc=$?"

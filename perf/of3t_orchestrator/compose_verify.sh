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

# Where this script lives -- the compose runs inside a scratch worktree, so a relative path to
# the sibling asserters resolves against the wrong tree.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# A row is listed here from the moment it is dispatched, not from its first push, so a new row
# cannot be silently left out of the composition. Rows with no branch yet are skipped with a line
# saying so -- silence would be the bug.
#
# That property was a COMMENT and not code until pass 175, and it was false when checked: this
# list stopped at `auxheads` while `conditioning`, `adaln`, `softmax` and `refprec` had all been
# dispatched AND pushed branches to origin. Four rows carrying the campaign's most recent
# measurements -- including the softmax NO-GO and the row that refuted D55's mechanism -- were
# absent from every composition, and the compose reported "18 of 18 rows" while doing it,
# because the denominator was the same stale string as the numerator. A hand-maintained list
# cannot enforce "listed from the moment it is dispatched"; the briefs are the record of what
# was dispatched, so derive it from them.
ROWS_FLOOR="reference tape equivalence data perf memory confidence leaves gradients pairbias l1 updaterule entity diffusion reopen rebase confhead auxheads conditioning adaln softmax trajectory"
WS="${WS:-/home/moritz/.coworker/workstreams}"
# A brief is "dispatched" when it carries a `#DISPATCH:` line -- the same fact the fleet queue
# reads to launch it, so this cannot disagree with what actually ran.
ROWS_SEEN="$( { for _b in "$WS"/of3t-*.txt; do
    [ -f "$_b" ] || continue
    grep -q '^#DISPATCH:' "$_b" || continue
    _n="$(basename "$_b" .txt)"; _n="${_n#of3t-}"
    [ "$_n" = "orchestrator" ] && continue   # the composition's author is not one of its rows
    printf '%s\n' "$_n"
  done; } | sort -u | paste -sd' ' - )"
# The floor is a RATCHET, not a default: a brief that is renamed or retired must not silently
# shrink the composition, so the union is what composes. A row present only in the floor is
# announced, because that means its brief stopped saying it was dispatched.
# ORDER MATTERS and pass 175 learned it the hard way. The first version of this derivation
# `sort -u`'d the union, which is deterministic but ALPHABETICAL -- and the floor's hand-written
# order was not arbitrary: it is the order rows were chartered, so an earlier row's version of a
# shared file lands before a later row's. Sorting alphabetically moved `refprec` ahead of
# `trajectory` and the compose hit a conflict in `perf/of3t_trajectory/agreement.py`, a file in
# trajectory's OWN namespace, because refprec's branch carried an older copy of it through a
# merge. So: the floor keeps its curated order, and rows known only from a brief are appended
# after it, sorted among themselves for determinism.
ROWS="$ROWS_FLOOR"
for _r in $ROWS_SEEN; do
  case " $ROWS_FLOOR " in *" $_r "*) ;; *) ROWS="$ROWS $_r" ;; esac
done

# HELD OUT, announced every run and never silent. A row lands here only when its branch cannot be
# composed with the others AT ALL -- not a conflict a resolver can take, but two independent
# implementations of one file -- and only after it has been told, in its brief, what to base on
# instead. The alternative is a composition that stops composing until a mid-flight row rebases,
# which hides every OTHER row's evidence behind one row's duplicate work.
#
#   d10d24-unify  pass 271. Wrote `tt_bio/ranking.py` and `tests/test_sample_ranking.py` from
#                 scratch; both already exist finished on `wk/of3t-rankunify`, which CONCLUDED and
#                 is what Moritz's 9629 decision means by "merge the unified rule". add/add, 7 and
#                 2 hunks. AMENDMENT 1 tells it to rebase onto rankunify and keep ITS files.
#   d56-renorm    pass 271. Edits `perf/of3t_orchestrator/assert_new_levers_default_off.py` --
#                 the instrument that checks its own lever, which is the one file a row may not
#                 change -- and conflicts there in 3 hunks plus 2 in tt_bio/ from a stale base.
#                 AMENDMENT 1 tells it to flip the default in its own namespace, say what the
#                 assert must become, and rebase onto d116's unified helper.
#   d1-pairbias   pass 272. Branched from main, so it conflicts in five tt_bio/ files at once, and
#                 it created `perf/of3t_pairbias/attn_f64.py` -- the concluded row of3t-pairbias's
#                 namespace -- add/add. Its brief now carries the STANDING base-on-wk/of3t and
#                 own-namespace rules that the whole 9629 dispatch wave was sent out without.
# RELEASED pass 273: d10d24-unify rebased onto wk/of3t and now merges clean (`git merge-tree`
# against the published composition), so the hold is lifted. A hold that outlives the thing it was
# for is the same rust the D149 ratchet refuses.
# RELEASED pass 274: d56-renorm unified the flag (autograd.SOFTMAX_BW_RENORM is now the single
# definition and taped_ttnn._SOFTMAX_BW_RENORM an alias -- D151 repaired), dropped its edit to
# assert_new_levers_default_off.py, and merges clean. That gate change is adopted here instead,
# in the same pass, so the assert and the shipped default move together.
# RELEASED pass 280: d1-pairbias CONCLUDED GO, and a concluded row's evidence must be in the
# composition rather than held outside it. Its three remaining conflicts are resolved by
# resolve_d1_pairbias.py, which takes the ROW's side on Moritz's ask-9629 ruling and records why.
HELD_OUT=""
for _h in $HELD_OUT; do
  _keep=""
  for _r in $ROWS; do [ "$_r" = "$_h" ] || _keep="$_keep $_r"; done
  ROWS="$(printf '%s' "$_keep" | sed 's/^ //')"
  echo "HELD OUT of3t-$_h: told to rebase (see its brief); NOT in this composition and NOT silently dropped"
done
for _r in $ROWS; do
  case " $ROWS_FLOOR " in *" $_r "*) _inf=1 ;; *) _inf=0 ;; esac
  case " $ROWS_SEEN " in *" $_r "*) _ins=1 ;; *) _ins=0 ;; esac
  [ "$_inf" = 0 ] && echo "  NOTE of3t-$_r: dispatched brief not in ROWS_FLOOR -- composing it"\
                          " from the brief. Add it to the floor once the row concludes."
  [ "$_ins" = 0 ] && echo "  NOTE of3t-$_r: in ROWS_FLOOR but its brief carries no #DISPATCH:"\
                          " line -- retired or renamed. Still composed; the floor is a ratchet."
done
unset _r _inf _ins _b _n
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
      elif [ "$_u" = "tt_bio/openfold3_trunk.py" ] && [ "$r" = "foldab" ]; then
        # The second file where a conflict is NOT a disagreement. Both sides INSERT around an
        # unchanged anchor line: of3t-pairbias documents why the trunk default stays False,
        # of3t-foldab adds an env-gated measurement lever after it and restates a shortened
        # copy of pairbias's comment. Keeping the lever + pairbias's FULL comment is what each
        # side meant. Unlike .gitignore this is shipped code, so the resolution is ASSERTED,
        # not trusted: assert_trunk_lever_resolution.py checks from the AST that pairbias's
        # unconditional default survived and that every env write to it sits inside a
        # not-None guard, with three negative controls behind it. Any other path still stops.
        python3 - <<'_RESOLVE'
p = "tt_bio/openfold3_trunk.py"
s = open(p).read()
i = s.index("<<<<<<< HEAD\n"); j = s.index("=======\n", i)
k = s.index(">>>>>>> origin/wk/of3t-foldab\n")
ours = s[i + len("<<<<<<< HEAD\n"):j]
theirs = s[j + len("=======\n"):k]
lever = theirs.split("        # scale_pair_bias=False:")[0]
open(p, "w").write(s[:i] + lever + ours + s[k + len(">>>>>>> origin/wk/of3t-foldab\n"):])
_RESOLVE
        python3 "$HERE/assert_trunk_lever_resolution.py" tt_bio/openfold3_trunk.py \
          || { echo "CONFLICT merging of3t-$r: openfold3_trunk.py resolution FAILED its assert"; exit 1; }
        python3 -m py_compile tt_bio/openfold3_trunk.py \
          || { echo "CONFLICT merging of3t-$r: resolved openfold3_trunk.py does not compile"; exit 1; }
        git add tt_bio/openfold3_trunk.py && git commit --no-edit -q
        echo "  NOTE of3t-$r: openfold3_trunk.py conflict resolved by keeping BOTH inserts"\
             " (pairbias comment + foldab env lever), asserted from the AST, not assumed"
      elif [ "$_u" = "tt_bio/openfold3_confidence.py" ] && [ "$r" = "auxfind" ]; then
        # Third file where a conflict is not a disagreement: two rows APPEND keyword arguments to
        # the same signature -- one `s_path`/`dtype`, of3t-auxfind `token_mask`/`single_mask` for
        # the reference-mask fix behind the aux_heads A18 failure. Both are optional, so the union
        # is what each side meant and no existing caller changes. Shipped code, so it is ASSERTED
        # from the AST: all four names present, each still with a default (a merge that dropped
        # one side would compile and import, and fail only at runtime on a device). Five negative
        # controls, including one that reorders a parameter into a SyntaxError.
        python3 - <<'_RESOLVE'
p = "tt_bio/openfold3_confidence.py"
s = open(p).read()
i = s.index("<<<<<<< HEAD\n"); j = s.index("=======\n", i)
k = s.index(">>>>>>> origin/wk/of3t-auxfind\n")
ours = s[i + len("<<<<<<< HEAD\n"):j].rstrip()
theirs = s[j + len("=======\n"):k].rstrip()
merged = ours[:-2].rstrip().rstrip(",") + ", " + theirs.strip()
open(p, "w").write(s[:i] + merged + "\n" + s[k + len(">>>>>>> origin/wk/of3t-auxfind\n"):])
_RESOLVE
        python3 "$HERE/assert_confidence_forward_signature.py" tt_bio/openfold3_confidence.py \
          || { echo "CONFLICT merging of3t-$r: openfold3_confidence.py resolution FAILED its assert"; exit 1; }
        git add tt_bio/openfold3_confidence.py && git commit --no-edit -q
        echo "  NOTE of3t-$r: openfold3_confidence.py signature conflict resolved by keeping BOTH"\
             " parameter sets, asserted from the AST with defaults intact"
      elif [ "$_u" = "tt_bio/openfold3_confidence.py" ] && [ "$r" = "confidence" ]; then
        # Fourth: same file, DIFFERENT rule, and the difference matters. main gained M18's
        # `tri_att_sdpa_hifi=...` at OF3's Pairformer-family sites on 2026-09-21; of3t-confidence
        # (concluded 09-19) carries `s_fp32_residual=True`. Those two are a union like auxfind's.
        # But the same hunk ALSO disagrees on `scale_pair_bias`: main ships **False**, the row's
        # branch carries **True**, and that is **D1** -- a repair that is HELD because applying it
        # measured 0.149 A WORSE at rank 0, and which pin 9629 asks Moritz to decide. A blind union
        # would take one of them arbitrarily; taking the row's would apply a held repair inside the
        # composition. So the rule is: union the NAMES, and on a collision **HEAD wins**, because
        # HEAD is main and main is what ships. Asserted below, by value, not just by presence.
        python3 - <<'_RESOLVE'
import re
p = "tt_bio/openfold3_confidence.py"
s = open(p).read()
i = s.index("<<<<<<< HEAD\n"); j = s.index("=======\n", i)
k = s.index(">>>>>>> origin/wk/of3t-confidence\n")
ours = s[i + len("<<<<<<< HEAD\n"):j].rstrip()
theirs = s[j + len("=======\n"):k].rstrip()
indent = re.match(r"\s*", ours).group(0)

def kwargs(text):
    """name -> full `name=value` source, splitting only at top-level commas.

    The trailing `)` closes the CALL and must come off before the depth counter runs, or it
    drives depth negative and every later top-level comma is missed -- which is exactly what
    the first version did: it dropped the last two kwargs and the closing paren, and the AST
    assert caught it on a SyntaxError rather than on a device.
    """
    t = text.rstrip()
    if not t.endswith(")"):
        raise SystemExit("resolution: a conflict side does not end the call with ')'")
    out, depth, cur = {}, 0, ""
    for ch in t[:-1].replace("\n", " ") + ",":
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            x = cur.strip()
            if "=" in x:
                out[x.split("=", 1)[0].strip()] = x
            cur = ""
        else:
            cur += ch
    return out

a, b = kwargs(ours), kwargs(theirs)
merged = dict(b); merged.update(a)          # HEAD (main) wins every collision
order = list(a) + [n for n in b if n not in a]
body = ",\n".join(indent + merged[n] for n in order) + ")"
open(p, "w").write(s[:i] + body + "\n" + s[k + len(">>>>>>> origin/wk/of3t-confidence\n"):])
_RESOLVE
        python3 - <<'_ASSERT' || { echo "CONFLICT merging of3t-$r: confidence resolution FAILED its assert"; exit 1; }
import ast, sys
src = open("tt_bio/openfold3_confidence.py").read()
tree = ast.parse(src)                                   # a lost bracket fails HERE, not on a device
need = {"scale_pair_bias", "fp32_softmax", "accurate_softmax",
        "tri_att_sdpa_hifi", "s_fp32_residual"}
found = {}
for n in ast.walk(tree):
    if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "Pairformer":
        for kw in n.keywords:
            if kw.arg in need:
                found[kw.arg] = ast.unparse(kw.value)
miss = sorted(need - set(found))
if miss:
    print("Pairformer(...) lost keyword(s):", ", ".join(miss)); sys.exit(1)
if found["scale_pair_bias"] != "False":
    print("scale_pair_bias resolved to", found["scale_pair_bias"],
          "-- main ships False and D1 is HELD (0.149 A worse at rank 0, pin 9629)"); sys.exit(1)
print("  confidence Pairformer keeps all five kwargs; scale_pair_bias=False as main ships")
_ASSERT
        git add tt_bio/openfold3_confidence.py && git commit --no-edit -q
        echo "  NOTE of3t-$r: confidence conflict resolved by UNION of names with HEAD winning"\
             " scale_pair_bias (D1 is HELD), asserted from the AST by VALUE"
      else
        # GENERIC LAST RESORT, and it exists because the per-row cases above do not scale. When
        # main touches a shared signature -- 2026-09-21, M18's `tri_att_sdpa_hifi` at OF3's four
        # Pairformer-family sites -- EVERY concluded row that had appended a kwarg at one of
        # those sites conflicts on the same shape at once. resolve_kwarg_tail_conflict.py takes
        # only conflicts where BOTH sides are pure keyword-argument tails, unions the names,
        # lets HEAD win any collision (main is what ships; at confidence the contested name is
        # `scale_pair_bias`, which is D1 and HELD), and refuses with exit 2 on anything else.
        # A `.gitignore` in the same merge is unioned first, since that rule is already settled.
        _left=""
        for _f in $_u; do
          if [ "$_f" = ".gitignore" ]; then
            git show :2:.gitignore > /tmp/.gi_ours 2>/dev/null
            git show :3:.gitignore > /tmp/.gi_theirs 2>/dev/null
            cat /tmp/.gi_ours /tmp/.gi_theirs | awk '!seen[$0]++ || $0==""' > .gitignore
            rm -f /tmp/.gi_ours /tmp/.gi_theirs
            git add .gitignore
          elif [ "$r" = "d1-pairbias" ] && "$PY" "$HERE/resolve_d1_pairbias.py" "$_f"; then
            # Two rows, opposite conclusions on the same lines, and not a stale base: of3t-pairbias
            # held the OF3 trunk default at False on nine seeds of 1UBQ; of3t-d1-pairbias flips it on
            # Moritz's ruling and 48 folds over four targets, with 1UBQ identified as the earlier
            # reading's own target. The resolver takes the row's side, keeps the superseded reasoning
            # in the file, and refuses if either goes missing.
            git add "$_f"
          elif [ "$r" = "d116" ] && "$PY" "$HERE/resolve_d116_softmax_inner.py" "$_f"; then
            # d116 unified the softmax-backward inner term across its two identical call sites
            # and is based on a main from before `_v_softmax` moved to the box pattern. Keep
            # HEAD's box read and `__all__`, take d116's helper call and its new name. The
            # resolver refuses the moment the hunk stops having that exact shape.
            git add "$_f"
          elif "$PY" "$HERE/resolve_prose_only_conflict.py" "$_f" "origin/wk/of3t-$r"; then
            # Both sides differ only in comments and docstrings -- a row based on an older
            # wk/of3t reflowed a comment, or carries a wording main has since sharpened. HEAD's
            # prose wins, and the safety is checked not argued: both sides are reconstructed,
            # parsed, stripped of docstrings and compared as ASTs, so any executable difference
            # anywhere refuses and stops the compose.
            git add "$_f"
          elif "$PY" "$HERE/resolve_softmax_inner_box.py" "$_f" "origin/wk/of3t-$r"; then
            # The BOX memory policy against D56's shared `softmax_bw_inner`, in either
            # orientation. Two repairs on the same three lines, neither aware of the other, and
            # they compose: read the handle through the box, then call the helper. Taking HEAD
            # alone leaves `y` unbound and the backward raises NameError the first time it runs,
            # which a collection-only compose cannot see. Generalises d116's literal, whose
            # orientation flipped the moment D56 landed on main.
            git add "$_f"
          elif "$PY" "$HERE/resolve_kwarg_tail_conflict.py" "$_f" "origin/wk/of3t-$r"; then
            git add "$_f"
          else
            _left="$_left $_f"
          fi
        done
        if [ -n "$_left" ]; then
          echo "CONFLICT merging of3t-$r, and these are not keyword-argument tails:"
          printf '  %s\n' $_left; exit 1
        fi
        git commit --no-edit -q
        echo "  NOTE of3t-$r: conflict(s) resolved by kwarg-tail UNION with HEAD winning"\
             " collisions; anything that was not a kwarg tail would have stopped the compose"
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
if ! git merge --no-edit -q wk/of3t-orchestrator; then
  # One conflict shape is resolvable here and exactly one: an add/add inside
  # `perf/of3t_orchestrator/`, the orchestrator's OWN artifact namespace. D171, pass 319. The
  # charter evaluator was written by `of3t-d122-d115` INTO this namespace and reached every tree
  # through that row's branch, never through the owner's, so when the owner finally carried it
  # the two copies had no merge base. Namespace ownership is the thing this campaign arbitrates
  # (`sibling-perf-campaigns-need-namespaced-output-paths`), so the owner's copy is the answer by
  # definition -- the other side is a row that wrote outside its own namespace.
  #
  # Narrow on purpose: ONLY add/add (both sides added, no base), ONLY under this one prefix, and
  # any other conflicted path still stops the composition. And it is self-verifying rather than
  # trusted -- everything under this prefix is an instrument the compose RUNS a few lines below,
  # so a resolution that dropped something load-bearing fails the run it is resolving. That is
  # the property, not the intention: `charter_evidence.py` refuses to publish if its own break
  # and negative controls do not pass.
  _u="$(git diff --name-only --diff-filter=U)"
  _bad="$(printf '%s\n' "$_u" | grep -v '^perf/of3t_orchestrator/' || true)"
  _notaa="$(for _f in $_u; do
              if git ls-files -u -- "$_f" | awk '{print $3}' | grep -qx 1; then
                printf '%s\n' "$_f"
              fi
            done; :)"
  if [ -n "$_u" ] && [ -z "$_bad" ] && [ -z "$_notaa" ]; then
    for _f in $_u; do
      # HEAD here is the accumulated composition and `wk/of3t-orchestrator` is what is being
      # merged, so the owner's copy is THEIRS, not OURS. Getting this backwards would silently
      # keep the row's stale copy and print the reassuring note anyway.
      git checkout --theirs -- "$_f" && git add -- "$_f"
      echo "  NOTE wk/of3t-orchestrator: add/add on $_f resolved to the NAMESPACE OWNER's copy"\
           " (D171); it is an instrument this compose runs, so a wrong resolution fails below"
    done
    git commit --no-edit -q
  else
    echo "CONFLICT merging wk/of3t-orchestrator:"; printf '%s\n' "$_u"
    [ -n "$_bad" ] && echo "  (outside perf/of3t_orchestrator/ -- not the D171 shape)"
    [ -n "$_notaa" ] && echo "  (has a merge base, so it is a real disagreement, not add/add:"\
                             " $_notaa)"
    exit 1
  fi
fi

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
#   tt_bio/openfold3_trunk.py: of3t-foldab owns the env-gated measurement lever
#   `TT_BIO_OF3_TRI_END_BIAS_FOLLOWS_PAIR`, which forces `tri_att_end_bias_follows_pair` (the
#   TRANSPOSE_BIAS orientation); of3t-trunkcliff owns the construction-site comment recording
#   what the PAIR-BIAS SCALE convention costs at the activation level. Two DIFFERENT flags on
#   adjacent lines, and unlike the entries above the hunks OVERLAP -- both start at line 132
#   (foldab +7 lines, trunkcliff +27), so disjointness is NOT the argument here.
#   The argument is that the merged result was CHECKED and carries both, which is asserted
#   below rather than declared: a silent pick of one side is the exact failure this list could
#   otherwise wave through (memory: parallel-branches-independently-fix-same-defect).
#   Semantic coupling, which a reader must not confuse: they are different conventions and they
#   do not interact -- of3t-trunkcliff measured the pair track BIT-IDENTICAL under its
#   convention change, and of3t-trunk043ref measured the single track unmoved across foldab's
#   orientation (0.101290 vs 0.101335). Both must stay; neither may flip a default.
#   perf/of3t_condtrans/floor_bf16.py: of3t-condtrans (CONCLUDED at b6cc90acc) owns the `f64`
#   policy and `--capture-ln`, which re-derive the REFERENCE operands at named LayerNorm sites;
#   of3t-cond043 owns `--expect-version`, which reads the package version off the imported
#   module's own directory and hard-fails a mismatch -- it exists because of3t_gradients/pylibs
#   carries an openfold3-0.5.0 dist-info that would make importlib.metadata report 0.5.0 behind a
#   0.4.3 source tree. The hunks OVERLAP: both edit the same argparse block and the same
#   docstring, so disjointness is NOT the argument. The argument is that both are wanted in ONE
#   instrument -- a private copy would fork the campaign's only diffusion-scope floor -- and that
#   the merged result is ASSERTED to carry both rather than assumed. Added pass 238, when
#   of3t-cond043 was still unpushed and the collision was still avoidable.
#   tt_bio/autograd.py: of3t-d116 owns `softmax_bw_inner`, the ONE expression it factored out of
#   `triangle_attention` and `_v_softmax` so the TT_BIO_SOFTMAX_BW_RENORM repair cannot be applied
#   to one of two identical sites -- which matters now that Moritz's 9629 decision is SHIP IT ON;
#   of3t-d137-tapegate owns `host_f64_softmax` / `host_f64_softmax_values` and the tape gate, the
#   D137 safety fix. The hunks OVERLAP in `__all__`, so disjointness is NOT the argument -- both
#   are wanted in the one tape, and the merged result is ASSERTED below to carry both rather than
#   assumed. Added pass 271. NOTE of3t-d116 is based on a main from before `_v_softmax` moved to
#   the box pattern; `resolve_d116_softmax_inner.py` bridges that and the row is told to rebase.
#   perf/of3t_trajwide/{trajwide,price_ref,ceiling,ceiling_closures}.py: of3t-trajwide owns the
#   measurement; of3t-refsweep owns the PATH LINES, by dispatch (D153) -- it repoints 43 scripts
#   in 11 namespaces off `/home/ttuser/of3t_rebase/`, which went with of3t-rebase's worktree, and
#   a sys.path entry that does not exist resolves nothing so `import openfold3` falls through to
#   0.5.0. A per-row sweep was not possible: the roots span namespaces whose rows have concluded.
#   The hunks are disjoint by construction -- the sweep's edit to each of these four files is the
#   three lines that move `refpath` from the row's directory to `perf/`, which it did because
#   eight namespaces import it now. Checked at pass 279: bit-identical tree behind the new path
#   (digest 1b27f5754b32b8e3 over 293 .py files, reproduced by a fresh pip download), so no number
#   moves. Asserted below rather than assumed.
ALLOWED_COEDIT="tt_bio/tenstorrent.py tt_bio/train/optim.py tt_bio/openfold3_trunk.py perf/of3t_condtrans/floor_bf16.py tt_bio/autograd.py perf/of3t_trajwide/trajwide.py perf/of3t_trajwide/price_ref.py perf/of3t_trajwide/ceiling.py perf/of3t_trajwide/ceiling_closures.py"
_coedit_floor=0

dup=$(awk '{print $2}' "$SLUG_TMP/own.txt" | sort | uniq -d)
for a in $ALLOWED_COEDIT; do
  if printf '%s\n' "$dup" | grep -qx "$a"; then
    printf 'ownership: %s co-edited by' "$a"
    grep " $a\$" "$SLUG_TMP/own.txt" | awk '{printf " %s",$1}'
    case "$a" in
      tt_bio/openfold3_trunk.py)
        # Do NOT claim disjointness here: these hunks OVERLAP (both start at line 132). What is
        # verified for this file is that the merged result carries both sides, asserted below.
        echo " -- DECLARED, hunks OVERLAP, both sides asserted present below" ;;
      perf/of3t_condtrans/floor_bf16.py)
        # Same shape: overlapping hunks in one argparse block, both sides asserted below. The
        # flag is set here rather than asserting unconditionally, because until of3t-cond043
        # pushes its branch only one side EXISTS and an unconditional assert would abort every
        # compose on a collision that has not happened yet.
        _coedit_floor=1
        echo " -- DECLARED, hunks OVERLAP, both sides asserted present below" ;;
      *)
        echo " -- DECLARED, regions verified disjoint" ;;
    esac
  fi
  dup=$(printf '%s\n' "$dup" | grep -vx "$a" || true)
done
dup=$(printf '%s\n' "$dup" | sed '/^$/d')
if [ -z "$dup" ]; then
  echo "ownership: no undeclared file is edited by more than one row"
  # The one co-edited file whose hunks OVERLAP: prove both sides survived, do not assume it.
  _tf="$CO/tt_bio/openfold3_trunk.py"
  _miss=""
  grep -q "TT_BIO_OF3_TRI_END_BIAS_FOLLOWS_PAIR" "$_tf" || _miss="$_miss of3t-foldab's env lever"
  grep -q "folds the bias inside its own score scale" "$_tf" || _miss="$_miss of3t-trunkcliff's pair-bias note"
  if [ -n "$_miss" ]; then
    echo "CO-EDIT LOST A SIDE in tt_bio/openfold3_trunk.py --$_miss"; exit 1
  fi
  echo "co-edit: openfold3_trunk.py carries BOTH foldab's lever and trunkcliff's pair-bias note"
  # The fourth: of3t-refsweep's path sweep over of3t-trajwide's live scripts. Prove the sweep
  # kept the row's substance rather than assuming a path-only diff stayed path-only.
  if printf '%s ' $PRESENT | grep -q "refsweep " && printf '%s ' $PRESENT | grep -q "trajwide "; then
    _tw="$CO/perf/of3t_trajwide/trajwide.py"
    _twmiss=""
    grep -q "def align_layer_norm_z" "$_tw" || _twmiss="$_twmiss trajwide's align_layer_norm_z"
    grep -q "assert_resolved" "$_tw" || _twmiss="$_twmiss trajwide's resolved-tree assertion (D149)"
    [ -f "$CO/perf/refpath.py" ] || _twmiss="$_twmiss the shared perf/refpath.py the sweep moved it to"
    if [ -n "$_twmiss" ]; then
      echo "CO-EDIT LOST A SIDE in perf/of3t_trajwide/trajwide.py --$_twmiss"; exit 1
    fi
    echo "co-edit: trajwide.py keeps its align_layer_norm_z and its D149 resolved-tree assertion after refsweep's path move"
  fi
  # The third overlapping co-edit, asserted only when both rows are in this composition.
  if printf '%s ' $PRESENT | grep -q "d116 " && printf '%s ' $PRESENT | grep -q "d137-tapegate "; then
    _af="$CO/tt_bio/autograd.py"
    _amiss=""
    grep -q "def softmax_bw_inner" "$_af" || _amiss="$_amiss of3t-d116's unified softmax_bw_inner"
    grep -q "SOFTMAX_BW_RENORM" "$_af" || _amiss="$_amiss the D56 renorm branch inside it"
    grep -q "host_f64_softmax" "$_af" || _amiss="$_amiss of3t-d137-tapegate's host_f64_softmax"
    if [ -n "$_amiss" ]; then
      echo "CO-EDIT LOST A SIDE in tt_bio/autograd.py --$_amiss"; exit 1
    fi
    echo "co-edit: autograd.py carries BOTH d116's unified softmax_bw_inner (with the D56 renorm) and d137-tapegate's host_f64_softmax"
  fi
  # The second overlapping co-edit, asserted only when both rows are actually in this composition.
  if [ "$_coedit_floor" = "1" ]; then
    _ff="$CO/perf/of3t_condtrans/floor_bf16.py"
    _fmiss=""
    grep -q -- "--capture-ln" "$_ff" || _fmiss="$_fmiss of3t-condtrans's --capture-ln"
    grep -q '"f64"' "$_ff" || _fmiss="$_fmiss of3t-condtrans's f64 policy"
    grep -q -- "--expect-version" "$_ff" || _fmiss="$_fmiss of3t-cond043's --expect-version"
    if [ -n "$_fmiss" ]; then
      echo "CO-EDIT LOST A SIDE in perf/of3t_condtrans/floor_bf16.py --$_fmiss"; exit 1
    fi
    echo "co-edit: floor_bf16.py carries BOTH condtrans's f64/--capture-ln and cond043's --expect-version"
  fi
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

# (3d) EVERY PUBLISHED HEADLINE RE-DERIVED FROM ITS OWN PER-TENSOR SIDECAR.
# recompute_from_sidecar.py was written at pass 175 with "exits non-zero if any --expect
# disagrees, so it can gate a compose" in its own docstring -- and was never wired in. It ran
# once, verified five headlines, and became a historical artifact. A check that ran once is not
# a guard: three headlines have changed since. This gates the four that cover 90.9251 % of the
# model's gradient mass, on CPU, with no device and no trust in any row's arithmetic -- only in
# its per-tensor diff_norm/ref_norm, which is why the campaign requires those instead of
# summary statistics. Controls run at pass 181: a wrong value and a missing section both exit 1.
echo "--- headlines re-derived from sidecars"
_RS="$HERE/recompute_from_sidecar.py"
_sc_fail=0
_sc() {  # <sidecar-path-in-CO> <section=value>
  [ -f "$CO/$1" ] || { echo "  SIDECAR MISSING: $1"; _sc_fail=1; return; }
  "$PY" "$_RS" "$CO/$1" --expect "$2" >/dev/null 2>&1 \
    || { echo "  HEADLINE DOES NOT RE-DERIVE: $1 expected $2"; _sc_fail=1; }
}
_sc perf/of3t_direct/sidecar_diffusion_conditioning/per_tensor_DEVICE_vs_UPSTREAM_BF16.json \
    "diffusion_module.diffusion_conditioning=0.06463839"
_sc perf/of3t_direct/sidecar_aux_heads/per_tensor_DEVICE_vs_UPSTREAM_BF16.json \
    "aux_heads=0.2360143"
_sc perf/of3t_residual/sidecar/per_tensor_DEVICE_vs_UPSTREAM_BF16.json "ALL=7.426217"
_sc perf/of3t_residual/sidecar/per_tensor_DEVICE_SOFTMAX_F64_BOUND_vs_UPSTREAM_BF16.json \
    "ALL=0.0777758"
[ "$_sc_fail" = 0 ] || { echo "COMPOSE: a published headline no longer re-derives from its sidecar"; exit 1; }
echo "  4 headlines re-derive from their per-tensor sidecars (90.9251 % of the gradient mass)"

# (4) the scoreboard against the artifacts. EVIDENCE.md is transcribed prose and a
# transcription drifts silently, so the numbers it quotes are re-read from the committed JSON
# on every compose. Also pins the denominators (K29).
# (3b) DISPATCH HYGIENE. `card=-` reads as "any TT card" to fleet.sh, never "none", and the
# failure is silent -- the row defers every two minutes with "no free card on any of [...]",
# which reads as capacity rather than a typo. Three live hits: 2026-08-22, 2026-09-07 (three
# days of deferrals), and 2026-09-20, mine, two rows at once. The memory entry asked twice for
# a check; this is it. Deliberately narrow -- see the script's SCOPE comment for the wider
# version that flagged 12 of 30 rows including one that plainly needed its card.
# (3c) a SUPERSEDED artifact must be NULLED, not merely stamped -- found unapplied to SEVEN of
# my own artifacts at pass 180, each still exposing structured number fields a reader or a
# script would consume as current. The stamp is documentation; the suffix is the interlock.
echo "--- superseded artifacts nulled"
"$PY" "$HERE/assert_superseded_is_nulled.py" || \
  { echo "COMPOSE: a superseded artifact still exposes live data fields"; exit 1; }

echo "--- dispatch card tokens"
"$PY" "$HERE/assert_dispatch_card_token.py" || \
  { echo "COMPOSE: a brief's #DISPATCH card token is wrong -- it will defer forever"; exit 1; }

# (3d) the audit's own published check COUNT is computed from confirmations, so any other
# guard that drifts lowers it and the count guard then blames "checks were added" -- the wrong
# cause, twice in one session at pass 236. This probe lifts that block out of the live audit
# and shows it refusing to evaluate while another check is down, while still firing on a
# genuinely stale count. CPU-only, no artifacts read.
# (3e) GO condition 5, priced. The plan for each USER-FACING defect is asserted against the live
# triage the gate reads, so it refuses rather than reporting a stale plan as a current one.
# (3f) INFERENCE MUST NOT REGRESS (Moritz, 2026-09-21, verbatim: "make sure regular inference is
# not changed to softmax fp64, not made slower. cause it was already in a good state. we did this
# only for training. i dont want to see regression in inference.")
#
# The fp32 softmax sites are on the SHARED triangle path -- af2.py, openfold3_trunk.py,
# openfold3_template.py, openfold3_msa_embedder.py and openfold3_confidence.py all set
# fp32_softmax=True -- and the composition already wires host_f64_softmax_site into Protenix as
# well as OpenFold3. So a default that reaches inference reaches EVERY model in tt-bio.
#
# The shipping line is main. This asserts it on every compose rather than once: no float64
# softmax symbol may exist on origin/main at all. Defaults-off in the composition is checked
# separately by assert_new_levers_default_off.py; this is the stronger, simpler property.
# (3g) D141. A capture's checkpoint provenance must prove BOTH halves of the load. The shared
# diffusion capture recorded missing_keys and not unexpected_keys, so 24 trained layer_norm_z
# tensors were dropped while the report read "1 missing, version_tensor". Three existing reports
# are frozen; a fourth must never ship blind.
# (3h) D142. PROTOCOL is read top to bottom by every row, so a clause conditioned on a defect
# that has since closed teaches a verdict the campaign no longer stands behind. Live conditions
# only -- ordinary provenance ("Record: D96") stays correct after a defect closes.
echo "--- PROTOCOL rests on no closed defect"
"$PY" "$HERE/assert_protocol_defect_refs.py" || \
  { echo "COMPOSE: a PROTOCOL clause carries a live condition on a defect that has closed"; exit 1; }

# (3h-bis) D166. Every closure, ratchet and GAP line in this campaign addresses a defect BY
# NUMBER, so two entries opening one D-number make a status word written for either read as the
# other's. Pass 308 did exactly that -- appended `### D165.` next to an existing `### D165.`,
# having picked the number off a `sort -n | tail` that printed 164. UPDATE headings are excluded
# deliberately: `### D164 UPDATE 1` is one defect, and the campaign uses that form on purpose.
echo "--- every defect D-number is opened exactly once"
"$PY" "$HERE/assert_defect_ids_unique.py" || \
  { echo "COMPOSE: two defect entries open the same D-number"; exit 1; }

# (3i) D148/A30. A summary of what the campaign still owes is composed from the state at the TOP
# of a pass, and rows report inside it -- DIRECTIVE-STATUS's "honest shape of what is left, at pass
# 199" was already wrong that same pass and stayed on the page for seventy more. The stamp must be
# present AND within ten passes; presence alone would have passed all seventy.
echo "--- summary paragraphs are stamped and fresh"
"$PY" "$HERE/assert_summary_stamped.py" || \
  { echo "COMPOSE: a 'what is left' summary is unstamped or more than ten passes stale -- see A30"; exit 1; }

echo "--- capture provenance records unexpected_keys"
( cd "$CO" && "$PY" perf/of3t_orchestrator/assert_capture_records_unexpected.py . ) || \
  { echo "COMPOSE: a capture report proves only half of its load -- see D141"; exit 1; }

# (3j) D149. `of3t-trajwide` ran openfold3 0.5.0 for its whole life while its constant and its
# prose said 0.4.3, because three `sys.path.insert(1, p)` calls reverse the order they were written
# to set. Narrow on purpose: the reversing LOOP, not the absence of a resolution read -- the broad
# version flagged 18 further files whose second insert is `os.getcwd()`.
# (3k) D151. One env var, two module-level reads, two backends -- and a comment asserting there
# is only one flag. Both default off today, so this is green when it lands; it goes red the moment
# a row flips one of the two, which is what of3t-d56-renorm's branch does.
# (3l) D122/D154. The gate that ends this campaign reads `state/of3t/CHARTER_EVIDENCE.json`.
# `audit_evidence.py` recomputes the four conditions against THIS tree and compares them to the
# copy inside the composition -- but the copy the GATE reads is the one in state/, and nothing
# re-derived it. They are byte-identical today only because one row wrote both. Publish it from
# the composition every pass, so the evaluation the gate reads is of `wk/of3t` and is never older
# than the compose that blessed it. The instrument REFUSES if its own break control fails, which
# is why this runs before the audit rather than after.
echo "--- publish the charter evaluation from the composition"
( cd "$CO" && "$PY" perf/of3t_orchestrator/charter/charter_evidence.py ) || \
  { echo "COMPOSE: charter_evidence.py refused to publish -- see its break control"; exit 1; }

# (3m) D155. A digest-equality claim that does not name its host cannot be attributed to healthy
# hardware, and pc card 0 is a faulty card root-caused 2026-08-17 that must not host bit-exact
# gating for any model at any size. Three of3t artifacts are in that state and are frozen; a new
# one fails. The list may only shrink.
# (3n) D162. Five per-site levers resolve at the construction site, so their shipped default is
# an ARGUMENT and not a module constant -- a census that walks flags to a resolved value has no
# row for the family at all. openfold3.trunk flipped to the fused HiFi SDPA path by default and
# the census was silent. assert_new_levers_default_off.py covers one of the five; this covers the
# rest by reading the call sites, as a shrink-only ratchet over the three that are ON today.
echo "--- site-flag call-site defaults (D162)"
( cd "$CO" && "$PY" perf/of3t_orchestrator/assert_site_flag_defaults.py . ) || \
  { echo "COMPOSE: a construction site ships a per-site lever ON by default, unpinned -- see D162"; exit 1; }

# (3o) D164. A guard that binds a sentence to its artifact cannot notice the artifact is the
# wrong INSTRUMENT. audit_evidence.py recomputed 172.43 + 698.32 = 870.75 and asserted cycles==1
# for 250 passes -- both true -- while the run those seconds came from was D14's MEMORY ladder,
# taking an allocator read on every one of 246,510 verb calls at 3.53 ms each. The same backward
# timed clean reads 222.48 s. Ask the question those assertions cannot: was it timing?
echo "--- a timing figure comes from a timing run (D164)"
( cd "$CO" && "$PY" perf/of3t_orchestrator/assert_timing_is_a_timing_run.py ) || \
  { echo "COMPOSE: live prose quotes a probe-instrumented run's wall clock as a timing figure -- see D164"; exit 1; }
( cd "$CO" && "$PY" perf/of3t_orchestrator/assert_timing_is_a_timing_run.py --self-test >/dev/null ) || \
  { echo "COMPOSE: the D164 guard's own negative control does not fire -- the guard is not a guard"; exit 1; }

echo "--- digest claims name their hardware (D155)"
( cd "$CO" && "$PY" perf/of3t_orchestrator/assert_digest_claims_name_their_card.py . ) || \
  { echo "COMPOSE: a digest claim cannot be attributed to healthy hardware -- see D155"; exit 1; }

echo "--- one flag, one default (D151)"
( cd "$CO" && "$PY" perf/of3t_orchestrator/assert_one_flag_one_default.py . ) || \
  { echo "COMPOSE: an env var's two readers disagree on its default -- see D151"; exit 1; }

echo "--- sys.path order: no new tree-resolution trust (D149)"
( cd "$CO" && "$PY" perf/of3t_orchestrator/assert_path_order_ratchet.py . ) || \
  { echo "COMPOSE: a new of3t script trusts a package-path constant instead of the resolution"; exit 1; }

echo "--- inference: no float64 softmax on main"
git fetch -q origin main 2>/dev/null || true
_f64_on_main=$(git grep -lE "host_f64_softmax|HOST_F64_SOFTMAX" origin/main -- tt_bio/ 2>/dev/null || true)
# Probe: the same grep must FIND the path on the composition, or this check is reading nothing
# and its silence on main means nothing (A17).
_f64_on_compose=$(git grep -lE "host_f64_softmax|HOST_F64_SOFTMAX" origin/wk/of3t -- tt_bio/ 2>/dev/null || true)
if [ -z "$_f64_on_compose" ]; then
  echo "COMPOSE: the float64-softmax grep finds nothing on wk/of3t either, so its silence on main"
  echo "         is uninformative -- the probe did not fire (A17)"; exit 1
fi
if [ -n "$_f64_on_main" ]; then
  echo "COMPOSE: a float64 softmax path is ON MAIN, which is the shipping line for every model in"
  echo "         tt-bio, not just OpenFold3 --"; printf '           %s\n' $_f64_on_main
  echo "         Moritz: \"i dont want to see regression in inference\". Revert it."; exit 1
fi
echo "  ok    no float64 softmax symbol on origin/main; the probe finds $(printf '%s\n' $_f64_on_compose | wc -l) file(s) on wk/of3t, so the grep reads something"

echo "--- user-facing closure plan"
"$PY" "$HERE/userfacing/closure_plan.py" | tail -4 || \
  { echo "COMPOSE: the USER-FACING closure plan is stale against UNFIXED_TRIAGE.json"; exit 1; }

echo "--- check-count evaluability"
"$PY" "$HERE/countstable/count_is_not_evaluable_while_drifted.py" | tail -2 || \
  { echo "COMPOSE: the check-count guard no longer refuses an unevaluable run"; exit 1; }

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
  # The header names the SOURCE THIS COPY WAS TAKEN FROM, derived from $_src rather than
  # re-typed from the loop variable. Until pass 307 it printed `state/of3t/$_f.md` for every
  # document including ORCHESTRATOR, whose source is special-cased one line above and whose
  # advertised path DOES NOT EXIST -- so the one instruction the header gives, "edit that one",
  # pointed a reader at nothing, or at a second live copy they would have had to create. That
  # is the defect the next two lines warn about, committed by the warning itself.
  { echo "<!-- PUBLISHED COPY, regenerated by compose_verify.sh on every compose."
    echo "     AUTHORITATIVE SOURCE: $_src on pc."
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
_PROTO="/home/moritz/.coworker/state/of3t/PROTOCOL.md"
_ORCH="/home/moritz/.coworker/state/of3t-orchestrator.md"
if [ -f "$_DOC" ] && grep -q '<!-- BEGIN GENERATED' "$_DOC"; then
  {
    echo "<!-- BEGIN GENERATED by compose_verify.sh -- do not edit between these markers -->"
    echo
    echo "Composed $(date -u +%F\ %TZ) from \`origin/main\` at \`$(git rev-parse --short origin/main)\`."
    echo "Head \`$(git -C "$CO" rev-parse --short HEAD)\`, **$(git -C "$CO" rev-list --count origin/main..HEAD) commits ahead**,"
    echo "carrying **$(set -- $PRESENT; echo $#) of $(set -- $ROWS; echo $#) rows**."
    echo
    # Pass 313. Three numbers in this file's AUTHORED prose had drifted: "seventeen amendments"
    # when PROTOCOL carries 30, "146 scoreboard figures", and -- the expensive one -- a closing
    # summary saying the model-dependent half was measured "at one Pairformer block, where it
    # FAILED, on a scope holding 0.20 % of the gradient's magnitude", written early and still
    # there with the campaign at 92.1568 %. The file's own opening paragraph warns about exactly
    # this class and then commits it three times, because it drew the authored/generated line in
    # the wrong place: it generated the TABLE and left the CLAIMS in prose. A claim with a number
    # in it is data. These three are derived here so they cannot drift again.
    echo "PROTOCOL carries **$(grep -oE '^\*\*A[0-9]+' "$_PROTO" 2>/dev/null | sort -u | wc -l | tr -d ' ') amendments**,"
    echo "each marked for whether a number already existed when it was written."
    echo
    echo "Campaign headline, copied verbatim from \`state/of3t-orchestrator.md\`'s \`VERDICT:\` so"
    echo "this file cannot state a different one:"
    echo
    sed -n 's/^VERDICT: /> **VERDICT:** /p' "$_ORCH" 2>/dev/null | head -1
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
# The SECOND shipped default the composition must not move, added pass 177. of3t-auxfind's
# confidence-mask fix is RELEASE-GATED and rides in the composition: it takes aux_heads' A18 from
# 4 of 5 heads at 5.174368e-01 to 0 of 5 at 3.865648e-03, and it would move pLDDT, PAE, PTM/IPTM
# and the ranking score on any padded fold. That is safe ONLY while it is off by default, and
# "off by default" is a property of the composed branch, not of the row's write-up -- so it is
# read from the tree. The asserter checks the default VALUE is None (a default merely EXISTING
# is not enough) and that the shipped fold path still does not pass the masks.
"$PY" "$HERE/assert_confidence_forward_signature.py" "$CO/tt_bio/openfold3_confidence.py" \
  || { echo "SHIPPED DEFAULT MOVED -- the release-gated confidence masks are live in the composition"; exit 1; }

# The THIRD and FOURTH shipped defaults, added pass 209 after both pass-207 repairs landed in the
# composition, and REVERSED for one of them at pass 274. TT_BIO_SOFTMAX_BW_RENORM now rides in LIVE
# on Moritz's ask-9629 ruling -- it moves every taped gradient (it is what takes the trunk from
# 9.025172e+00 to 3.833066e-01) and it cannot move a fold, because every read of it is inside a
# backward closure, checked by AST rather than asserted. The host float64 softmax still may not: it
# moves fold output and costs a round trip at any site where it is on, and D137 says it is gated on
# an env flag rather than on the tape. So the asserter pins BOTH directions -- renorm ON, host path
# OFF -- and for the host path checks both halves, the default AND that no construction site
# overrides it to True, because a selector defaulting False says nothing when a site passes
# default=True, which is exactly how opendde.refiner ships the accurate-softmax chain ON.
"$PY" "$HERE/assert_new_levers_default_off.py" "$CO" \
  || { echo "SHIPPED DEFAULT MOVED -- a pass-207 repair is live in the composition"; exit 1; }

_trunk="$CO/tt_bio/openfold3_trunk.py"
# Flipped at pass 280, in the same commit that lands the flip, which is what the previous version
# of this block instructed: "If Moritz approves it anyway, change _want in this script in the same
# commit that flips the default." He approved it (ask 9629) and of3t-d1-pairbias is GO.
_want='scale_pair_bias=True, tri_att_scale_pair_bias=False'
if grep -q "$_want" "$_trunk"; then
  echo "shipped defaults: OF3 trunk pair-bias is scale_pair_bias=True (ask 9629 DECIDED: fix it everywhere; of3t-d1-pairbias GO -- 48 folds, 4 targets, 6 seeds, sign test p=0.541, pLDDT up in 24 of 24; unmerged to main)"
else
  echo "SHIPPED DEFAULT MOVED -- $_trunk does not carry: $_want"
  grep -n 'scale_pair_bias=' "$_trunk" | sed 's/^/  /'
  echo "  D1 is DECIDED, not held: ask 9629 says fix it everywhere, and of3t-d1-pairbias measured"
  echo "  the reopen condition Moritz set -- 'reliably worse across targets and seeds' -- and did"
  echo "  not meet it: 10 of 24 paired folds regress, sign test p = 0.541, pooled median negative,"
  echo "  and the sole regressing target is 1UBQ, which is the target the 0.149 A reading was built"
  echo "  on. A revert to False is now the drift. If it is deliberate, say why here in the same"
  echo "  commit, and reopen the ask rather than moving the default quietly."
  exit 1
fi

# The composition is rebuilt from origin/main every pass, so its history is NEW each time and is
# never a fast-forward of the wk/of3t already on origin -- a plain push is rejected with
# "tip of your current branch is behind", which reads like a stale checkout and is not one.
# `--force-with-lease` against the SHA this script just read is the correct publish: it replaces
# the recomposition and still refuses if a neighbour pushed wk/of3t since.
_lease=$(git -C "$CO" rev-parse --verify -q origin/wk/of3t 2>/dev/null || echo "")
echo; echo "composition ready at $CO ; push with:"
if [ -n "$_lease" ]; then
  echo "  git -C $CO push --force-with-lease=wk/of3t:$_lease origin wk/of3t"
else
  echo "  git -C $CO push origin wk/of3t   # wk/of3t does not exist on origin yet"
fi
# `--push` exists because I once ran `compose_verify.sh; git -C $CO push -f` as one line and
# force-pushed a FAILED, half-merged composition over a good one: 256 commits replaced by 55.
# The branch is regenerated every pass so nothing was lost, but the shape of the mistake is
# permanent -- a push that is not conditional on the verdict is not a verified push.
if [ "${1:-}" = "--push" ]; then
  git -C "$CO" push -f -q origin wk/of3t \
    && echo "pushed wk/of3t -> $(git -C "$CO" rev-parse --short HEAD), \
$(git -C "$CO" rev-list --count origin/main..HEAD) ahead"
fi

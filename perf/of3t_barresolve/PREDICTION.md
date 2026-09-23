# PREDICTION — of3t-barresolve, registered before the first run

Written and pushed before `resolve_check.py` was run once. What follows is what I expect and
why, so the run can contradict it.

## The question

`of3t-trajbar` recorded `"ref_tree": TW.REF_TREE` in its steplogs and the composition's D149
guard refuses the file, on the reading that `TW.REF_TREE` is a named constant and nothing in
`trajbar.py` reads `openfold3.__file__` back. If the bar's reference side ran 0.5.0 the way
`of3t-trajwide`'s did before D149 was found, the 1.762065e-01 at k = 20 that GO condition 3
divides by is measured on the wrong upstream.

## What I predict, and the reason

**Outcome 1: it resolves 0.4.3 and the bar stands at 1.762065e-01.**

`TW.REF_TREE` is not a constant. `perf/of3t_trajwide/trajwide.py:52` initialises it to `None`
and `build_theirs` (line 294-298) assigns `REF_TREE = refpath.assert_resolved()` before it
builds anything. `trajbar.install_mixed` patches `build_theirs`, but `build_mixed` calls the
original, so the assert runs on the bf16-mixed arm too. `assert_resolved()` raises `SystemExit`
on a mismatch, so a wrong tree could not have produced a twenty-rung steplog at rc=0. The
committed steplogs already carry `/home/ttuser/of3t_refprec/of3pkg043`, 761 parameters, 0
missing and 0 unexpected, which is 0.4.3's signature — 0.5.0 loads this checkpoint at 1 missing
and 24 unexpected.

So the defect the guard names is real but it is a STATIC one: `trajbar.py` itself records a
value it never reads back, and a future edit that reaches `write_steplog` without going through
`build_theirs` would record `null` or a stale path with nothing refusing.

**Two trees, and I predict the run prints different paths on the two trees.** On this branch
(`wk/of3t-trajbar`) `refpath.py` is `perf/of3t_trajwide/refpath.py` with
`OF3PKG = /home/ttuser/of3t_refprec/of3pkg043`. In the composition, `of3t-refsweep` moved the
module to `perf/refpath.py` and repointed it at `REFROOT = /home/ttuser/of3t-campaign-refs`.
Both trees digest to `1b27f5754b32b8e3…` over 293 `.py` files, so I expect the same code and no
change to any number — but the path in the steplog and the path the next run gets are not the
same string, and that is worth stating rather than discovering later.

**Predicted RESOLVED line, branch tree:**
`REF_TREE resolved: /home/ttuser/of3t_refprec/of3pkg043`, 761 parameters, 0 missing, 0 unexpected.

**Predicted RESOLVED line, composition:**
`REF_TREE resolved: /home/ttuser/of3t-campaign-refs/of3pkg043`, same 761 / 0 / 0.

## What would refute it

Any resolution whose realpath is not an `of3pkg043` tree — `pylibs`, `of3pkg050`, a venv
site-packages — or a parameter count other than 761, or a non-zero missing/unexpected count.
Then the bar is on the wrong upstream, GO condition 3 loses its denominator, and this row stops
and reports instead of re-running anything.

## Confidence

High on outcome 1 (the assert is already on the path and would have refused), lower on the
two-path split being noticed as a finding rather than a footnote.

"""Where the OpenFold3 campaign's reference trees and captures live, for every row.

The reference is upstream **0.4.3** (`of3pkg043`), the tree `grads_f64_043.pt`, the block
boundaries and `SCOPE_LADDER.json` were taken on. 0.5.0 sits beside it inside `pylibs`, and at
this boundary the two are a different FUNCTION rather than a different rounding: D120 measured
1.94959719e-05 (0.5.0) against 7.66979728e-01 (0.4.3), f32 against f32.

Two ways that has already gone wrong, and one module answers both:

* D149 -- five scripts installed the trees with `sys.path.insert(1, p)` in a loop. Three inserts
  at the SAME index reverse the order, `pylibs` landed ahead of `of3pkg043`, and the reference
  side measured 0.5.0 while every artifact named 0.4.3.
* D153 -- 43 scripts across eleven namespaces hard-coded `/home/ttuser/of3t_rebase/...`, which
  `worker.sh` removed with its row's worktree. Prepending a path that does not exist resolves
  nothing, so `import openfold3` fell through to 0.5.0 in `pylibs` and printed no warning.

Order is not evidence and a named constant is not a resolution. `install()` APPENDS the
dependency trees and prepends only the tree under test, which is the convention
`perf/of3t_foldab/convention.py` states outright; `assert_resolved()` then imports the package,
reads `openfold3.__file__` and refuses with the resolved path. Its return value is what every
artifact records, per A24: a reference is an input.

`of3t-campaign-refs/` is where D112 restored and manifested these after the prune, in paths no
`rm` in `worker.sh` or `disk_guard.sh` reaches. Its `of3pkg043` is bit-identical to the copy
under `of3t_refprec/`: both digest to 1b27f5754b32b8e3... over 293 `.py` files under the
A24-AMENDMENT rule, which is also the value a fresh `pip download openfold3==0.4.3` reproduced.

The dep trees are NOT under `of3t-campaign-refs`. They still sit under `of3t_gradients/`, a
concluded slug's directory, which is the same exposure D112 closed for the package trees; they
are routed through `REFDEPS` here so moving them is a one-line change. See
`perf/of3t_refsweep/STATE_of3t-refsweep.md`.
"""
from __future__ import annotations

import os
import sys

REFROOT = "/home/ttuser/of3t-campaign-refs"

#: upstream 0.4.3, the tree every reference-side number in this campaign is taken on
OF3PKG = f"{REFROOT}/of3pkg043"
#: upstream 0.5.0, for the arms that deliberately price the version difference
OF3PKG050 = f"{REFROOT}/of3pkg050"
#: openfold3's own dependencies. `pylibs` also carries openfold3 0.5.0 -- that is the tree a
#: missing `OF3PKG` silently falls through to, and the reason `assert_resolved()` exists.
REFDEPS = ("/home/ttuser/of3t_gradients/deps", "/home/ttuser/of3t_gradients/pylibs")
#: `bundle_min.py`, the bundle loader the capture scripts import
REFCODE = "/home/ttuser/of3t_gradients/ref"

#: the float64 gradient bundle: grads_f64_043.pt, w0_043.pt, batch_step003.pt, draws, MANIFEST
BUNDLE = f"{REFROOT}/bundle_min_043"
#: trunk block boundaries at 0.4.3: block{0,23,47}_boundary.pt
CAP = f"{REFROOT}/cap"
#: the seven-rung ladder, block{0,8,16,23,32,40,47}_boundary.pt. Byte-identical to CAP on the
#: three blocks they share (sha256 21e10e3d... on block0, a55ef1c4... on block47), so this is
#: the same capture with four more rungs. Not under REFROOT; see the module docstring.
CAP_LADDER = "/home/ttuser/of3t_trunkdepth/cap_ladder"
#: the 0.4.3 diffusion boundary, rebuilt and triple-checked by of3t-softgrad (`recap043.sh`)
#: after the original was pruned. Not under REFROOT; see the module docstring.
DIFFCAP = "/home/ttuser/of3t_softgrad/diffcap043"


def install(pkg: str = OF3PKG, deps=REFDEPS) -> str:
    """Put `pkg` ahead of everything and the dep trees behind it. Idempotent."""
    for p in deps:
        if p not in sys.path:
            sys.path.append(p)
    while pkg in sys.path:
        sys.path.remove(pkg)
    sys.path.insert(0, pkg)
    return pkg


def assert_resolved(pkg: str = OF3PKG) -> str:
    """Import `openfold3` and return the tree it actually came from, or refuse."""
    import openfold3
    f = getattr(openfold3, "__file__", None)
    if not f:
        raise SystemExit(f"openfold3 has no __file__ (namespace package?); "
                         f"__path__ = {list(getattr(openfold3, '__path__', []))}")
    got = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(f))))
    want = os.path.realpath(pkg)
    if got != want:
        raise SystemExit(f"openfold3 resolved from {got}, not the tree under test {want} "
                         f"(openfold3.__file__ = {f}). Reference-side numbers taken here "
                         f"would be against the wrong upstream version -- see D149, D153.")
    return got


def require(*paths: str) -> None:
    """Refuse up front for every named input that is not there.

    D153: a missing reference path is invisible. `sys.path` ignores it, `PYTHONPATH` ignores
    it, and the run continues against whatever else answers to the name. Name the file instead.
    """
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("reference inputs missing: " + ", ".join(missing) +
                         " -- see perf/refpath.py for where the campaign's references live.")


if __name__ == "__main__":
    # `--path-only` proves the PYTHONPATH the CALLER already set, without touching it. That is
    # the shell case: `perf/refpath.sh` composes the path and this is what checks it.
    pkg = OF3PKG
    if "--pkg" in sys.argv:
        pkg = sys.argv[sys.argv.index("--pkg") + 1]
    if "--path-only" not in sys.argv:
        install(pkg)
    print(f"REF_TREE resolved: {assert_resolved(pkg)}")

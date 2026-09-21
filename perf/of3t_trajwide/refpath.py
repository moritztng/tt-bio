"""Where `openfold3` resolves from, for every script in this row.

The reference is upstream **0.4.3** (`of3pkg043`), the tree `grads_f64_043.pt` and
`SCOPE_LADDER.json` were taken on. 0.5.0 sits beside it in `pylibs`, and at this boundary the
two are a different FUNCTION rather than a different rounding: D120 measured 1.94959719e-05
(0.5.0) against 7.66979728e-01 (0.4.3), f32 against f32.

Until 2026-09-21 all five scripts here installed the trees with `sys.path.insert(1, p)` in a
loop over `(OF3PKG, deps, pylibs)`. Three inserts at the SAME index reverse the order: pylibs
goes in last and lands at index 1, ahead of of3pkg043 at index 3. So `openfold3` resolved from
0.5.0 while every artifact named 0.4.3, and the reference side of the trajectory measured a
different model than it reported (D149).

`install()` APPENDS the dependency trees and prepends only the tree under test, which is the
convention `perf/of3t_foldab/convention.py` states outright. `of3pkg043` carries `openfold3`
and `PKG-INFO` and nothing else, so prepending it shadows nothing; the deps it needs
(ml_collections, absl, pytorch_lightning, torchmetrics) come from `deps` and `pylibs` behind it.

A named constant is not a resolution -- that was the whole defect. `assert_resolved()` imports
the package, reads `openfold3.__file__`, and refuses with the resolved path in the message. Its
return value is what every artifact records, per A24: a reference is an input.
"""
from __future__ import annotations

import os
import sys

OF3PKG = "/home/ttuser/of3t_refprec/of3pkg043"
REFDEPS = ("/home/ttuser/of3t_refprec/deps", "/home/ttuser/of3t_refprec/pylibs")


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
                         f"would be against the wrong upstream version -- see D149.")
    return got

#!/usr/bin/env python3
"""of3t-cotterm: put the namespaces on sys.path in the order written, and read back what answered.

Two defects, one file (D149).

1. A loop that inserts N paths at ONE CONSTANT index reverses them. `armapb.py` and `refapb.py`
   both wrote the repo root first, meaning "prefer this tree", and both got it last of five.
   `install()` splices the list once, so the order written is the order you get.
2. A path constant is a request, not a resolution. Four sibling `perf/of3t_*` namespaces ahead
   of the repo root is exactly the setup where a same-named helper answers from the wrong one,
   and the numbers then look plausible. `resolved()` imports and reads `openfold3.__file__` and
   `tt_bio.__file__` BACK, refuses on a `require` pair that does not match, and returns a dict
   the caller puts in its artifact -- so a reader sees which trees produced the numbers.

`of3t-refsweep` and `perf/of3t_orchestrator/revision/capture_trunk_entry.py` are the pattern;
this is the same thing shared by both of this row's scripts rather than copied into each.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: the namespaces this row imports from, in PRECEDENCE order. The repo root is first because
#: `tt_bio` must come from the worktree under test and from nowhere else.
NAMESPACES = ("", "perf/of3t_apbleaf", "perf/of3t_trunkg043", "perf/of3t_gradients",
              "perf/of3t_cotterm")


def install(namespaces=NAMESPACES, root=ROOT):
    """Prepend `namespaces` to sys.path in the order written. Returns the paths, highest first."""
    paths = [os.path.join(root, p) if p else root for p in namespaces]
    for p in paths:
        while p in sys.path:
            sys.path.remove(p)
    sys.path[:0] = paths
    return paths


def resolved(require=(), modules=("apbmath", "armln", "refln", "dev_grad", "ref_grad")):
    """What actually answered each import. `require` is (module_name, prefix) pairs and is enforced.

    Reads `openfold3.__file__` and `tt_bio.__file__` back rather than trusting the constant that
    asked for them; a module that is not imported is reported as such rather than forced.
    """
    out = {"root": ROOT, "sys_path_head": list(sys.path[:len(NAMESPACES) + 1])}
    for name in ("openfold3", "tt_bio"):
        mod = sys.modules.get(name)
        out[f"{name}_file"] = getattr(mod, "__file__", None) if mod else "not imported"
    for name in modules:
        mod = sys.modules.get(name)
        if mod is not None:
            out[f"{name}_file"] = getattr(mod, "__file__", None)
    for name, prefix in require:
        mod = sys.modules.get(name)
        got = getattr(mod, "__file__", None) if mod else None
        if got is None:
            raise SystemExit(f"D149: {name} was never imported, so its tree cannot be read back")
        if not os.path.realpath(got).startswith(os.path.realpath(prefix)):
            raise SystemExit(f"D149: {name} resolved to {got}, which is not under {prefix}")
    return out


if __name__ == "__main__":
    import json
    install()
    import tt_bio                                                          # noqa: F401
    print(json.dumps(resolved(require=[("tt_bio", ROOT)]), indent=2))

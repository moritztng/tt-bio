#!/usr/bin/env python3
"""Resolve `of3t-d116`'s softmax-backward conflict: HEAD's box semantics, d116's unified helper.

`of3t-d116` factored `sum_j g_j y_j` out of two identical expressions -- `triangle_attention` and
`_v_softmax` -- into `autograd.softmax_bw_inner`, keeping the `TT_BIO_SOFTMAX_BW_RENORM` branch
inside it. Its docstring has the argument: *"the defect is the rule, not the site, and a repair
applied to one of two identical expressions is the kind of half-fix that reads as fixed."* That is
right, and D56 was just decided SHIP IT ON, so the branch must survive.

It conflicts because the row is based on a main from BEFORE `_v_softmax` moved to the box pattern.
HEAD reads `y = box[0]` inside the closure so `free` may evict y to DRAM and sets `out.box = box`;
d116 still closes over `y` directly and sets `out.evictable = False`. Those are the same tensor and
two different memory policies, and HEAD's is the later one.

So: HEAD's structure, d116's call. Both sides' intent survives and neither is silently dropped.
The proper fix is for the row to REBASE -- this is a bridge while it is mid-flight, and it refuses
the moment the conflict stops having the exact shape below, which is how it stops being a way to
paper over the next difference.

Usage: resolve_d116_softmax_inner.py <file>   ->  0 resolved, 2 not this shape (caller must stop).
"""
from __future__ import annotations

import ast
import pathlib
import sys

TAPED_OLD = """<<<<<<< HEAD
            y = box[0]                      # through the box, so `free` may evict y to DRAM
            inner = ttnn.sum(ttnn.multiply(g, y), dim=dim, keepdim=True)
            if _SOFTMAX_BW_RENORM:
                inner = ttnn.divide(inner, ttnn.sum(
                    y, dim=dim, keepdim=True, compute_kernel_config=precise_config()))
=======
            inner = ag.softmax_bw_inner(y, g, dim=dim)
>>>>>>> origin/wk/of3t-d116
"""
TAPED_NEW = """            y = box[0]                      # through the box, so `free` may evict y to DRAM
            # `of3t-d116` unified this expression with `triangle_attention`'s copy; the helper
            # carries the same TT_BIO_SOFTMAX_BW_RENORM branch this site used to inline.
            inner = ag.softmax_bw_inner(y, g, dim=dim)
"""

ALL_OLD = """<<<<<<< HEAD
    "Tensor", "precise_config", "no_grad", "parameter", "forget_parameters",
    "parameter_for", "untaped",
=======
    "Tensor", "precise_config", "softmax_bw_inner", "no_grad", "parameter",
    "forget_parameters",
>>>>>>> origin/wk/of3t-d116
"""
ALL_NEW = """    "Tensor", "precise_config", "softmax_bw_inner", "no_grad", "parameter",
    "forget_parameters", "parameter_for", "untaped",
"""

CASES = {"tt_bio/taped_ttnn.py": (TAPED_OLD, TAPED_NEW),
         "tt_bio/autograd.py": (ALL_OLD, ALL_NEW)}


def main(argv):
    if len(argv) < 2:
        print("usage: resolve_d116_softmax_inner.py <file>", file=sys.stderr)
        return 2
    path = pathlib.Path(argv[1])
    key = next((k for k in CASES if str(path).endswith(k)), None)
    if key is None:
        return 2
    old, new = CASES[key]
    text = path.read_text()
    if old not in text:
        print(f"{path}: d116 conflict is not the shape this resolver knows -- the row has moved, "
              f"rebase it rather than widening this", file=sys.stderr)
        return 2
    out = text.replace(old, new, 1)
    if "<<<<<<<" in out or ">>>>>>>" in out:
        print(f"{path}: a second conflict hunk remains after resolving the known one",
              file=sys.stderr)
        return 2
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"{path}: resolution does not parse ({e})", file=sys.stderr)
        return 2
    # Both intents must be present afterwards, asserted rather than assumed.
    if key.endswith("taped_ttnn.py"):
        if "box[0]" not in out or "softmax_bw_inner" not in out:
            print(f"{path}: resolution lost one side", file=sys.stderr)
            return 2
    else:
        if '"softmax_bw_inner"' not in out or '"parameter_for"' not in out:
            print(f"{path}: __all__ union lost a name", file=sys.stderr)
            return 2
    path.write_text(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

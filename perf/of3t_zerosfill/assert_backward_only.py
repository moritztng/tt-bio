#!/usr/bin/env python3
"""A46 clause 4, checked on the tree rather than taken on this row's word.

Moritz, 2026-09-21: *"make sure regular inference is not changed ... i dont want to see
regression in inference."* That is a hard stop. This row changed where a zero tensor is built,
and `tt_bio/autograd.py` and `tt_bio/taped_ttnn.py` are shared by every model in tt-bio, so the
claim "backward only" has to be checked, not asserted.

By AST: every read of `grad_zeros` and `DEVICE_ZEROS` must sit inside the tape's gradient
machinery, and so must every call site of each helper that reaches them. The terminal scopes are
named here rather than inferred:

  `bw`        every taped backward closure in `taped_ttnn.py` is named this
  `grad`      `Tensor.grad`, which joins the slice parts. `_parts` is populated only by
              `add_grad_slice`, which a forward never calls, so reading `.grad` on an untaped
              run returns `_grad` and touches nothing here
  `backward`  the replay itself

and the helper chain `grad_zeros <- _zeros_like_along <- _pad_slice <- _join_slices` is walked,
each link's own call sites checked the same way, so the transitive claim closes instead of being
asserted at the first hop. A module-level ASSIGNMENT to `DEVICE_ZEROS` is not a read and is
skipped by store context, not by line number.

    python3 perf/of3t_zerosfill/assert_backward_only.py
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
FILES = ["tt_bio/autograd.py", "tt_bio/taped_ttnn.py"]
NAMES = {"grad_zeros", "DEVICE_ZEROS"}

# Backward-only helpers: a function that exists to serve a `bw` and is reached from nowhere
# else. Each is verified below by checking its OWN call sites the same way, so naming one here
# is a claim this script then tests rather than an exemption.
HELPERS = {"grad_zeros", "_zeros_like_along", "_pad_slice", "_join_slices"}
TERMINAL = {"bw", "grad", "backward"}


def enclosing(tree):
    """{node: [enclosing function names, outermost first]}"""
    out, stack = {}, []

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, n):
            stack.append(n.name)
            for c in ast.iter_child_nodes(n):
                self.visit(c)
            stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def generic_visit(self, n):
            out[n] = list(stack)
            super().generic_visit(n)

    V().visit(tree)
    return out


def main() -> int:
    bad, checked = [], 0
    helper_sites = {h: [] for h in HELPERS}
    for rel in FILES:
        p = ROOT / rel
        tree = ast.parse(p.read_text())
        enc = enclosing(tree)
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Name) and node.id in NAMES | HELPERS:
                # A store is a definition, not a read: `DEVICE_ZEROS = True` at module level
                # is the flag's own default and says nothing about who reaches it.
                if isinstance(getattr(node, "ctx", None), ast.Store):
                    continue
                name = node.id
            elif isinstance(node, ast.Attribute) and node.attr in NAMES | HELPERS:
                name = node.attr
            if name is None:
                continue
            scope = enc.get(node, [])
            if name in HELPERS and scope and scope[-1] == name:
                continue                                   # its own definition
            if name in ("grad_zeros", "DEVICE_ZEROS") and scope and scope[0] == "grad_zeros":
                continue                                   # inside the one definition
            if name in HELPERS:
                helper_sites[name].append((rel, node.lineno, scope))
            if name not in NAMES:
                continue
            checked += 1
            ok = bool(TERMINAL & set(scope)) or any(s in HELPERS for s in scope)
            if not ok:
                bad.append(f"{rel}:{node.lineno} reads {name} outside a backward closure, "
                           f"scope={scope}")

    # The helpers' own call sites, so the transitive claim closes rather than being asserted.
    for h, sites in helper_sites.items():
        for rel, line, scope in sites:
            if not scope:
                continue
            if TERMINAL & set(scope) or any(s in HELPERS for s in scope):
                continue
            bad.append(f"{rel}:{line} calls backward-only helper {h} outside a backward "
                       f"closure, scope={scope}")

    print(f"checked {checked} reads of {sorted(NAMES)} across {len(FILES)} files; "
          f"terminal scopes {sorted(TERMINAL)}")
    for h, sites in sorted(helper_sites.items()):
        print(f"  {h}: {len(sites)} call sites, scopes "
              f"{sorted({s[-1] if s else '<module>' for _r, _l, s in sites})}")
    if bad:
        print("\nNOT backward-only:")
        for b in bad:
            print("  " + b)
        return 1
    print("\nbackward only: no forward or inference path reaches the changed zero fill")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

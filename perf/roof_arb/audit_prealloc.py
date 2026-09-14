#!/usr/bin/env python3
"""Every `ttnn.allocate_tensor_on_device` call site in `tt_bio`: is the buffer a WRITE or a READ?

The byte counter behind the campaign's 2.9449 TB cannot tell a pre-allocated buffer's writer from
its readers, because ttnn hands the destination to `ttnn.generic_op` in the same flat tensor list as
the operands. `roof-redteam-2` called it undecidable and bracketed one trimul block at 1745.6 or
2550.9 MB, 1.46x apart. It is decidable from the source, and this file decides it exhaustively
rather than by spot check: one counter-example changes the answer.

For each site it reports the variable, the helper the buffer is handed to, the argument slot, and
whether the value is returned. `parse` walks the AST, so a site cannot be missed by a grep that
does not match its formatting. The classification itself is a read of the helper, recorded in
HELPERS below with the line that proves it.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "tt_bio"

# helper -> (which parameter carries the destination, the line that proves it)
HELPERS = {
    "G.generic_minimal_matmul": ("outs, positional 4 of (device, in0, in1, outs, ...)",
                                 "tt_bio/mm_generic.py:372 ttnn.generic_op([in0, in1, *outs], pd)"),
    "SG.sdpa": ("out, positional 6",
                "tt_bio/triatt_sdpa.py:283 SG.sdpa(dev, q, k, v, bias, out, ...)"),
    "ttnn.generic_op": ("last element of the io tensor list",
                        "tt_bio/trimul_tail.py:280, rfd3_bias.py:224/440/641, "
                        "reblock_permute.py:254/497/754"),
    "_reblock.reblock_permute_gated": ("out=, keyword",
                                       "tt_bio/reblock_permute.py:754 generic_op([xw, out], pd)"),
    "ttnn.fill_cache": ("cache, positional 1, through ttnn.experimental.view",
                        "tt_bio/esmc.py:761 ttnn.fill_cache(view, out, i)"),
    "ttnn.copy_host_to_device_tensor": ("dst, positional 2",
                                        "tt_bio/rfd3/model.py:2342 and three siblings"),
}


ALIASING = {"ttnn.experimental.view", "ttnn.reshape", "ttnn.unsqueeze", "ttnn.squeeze"}


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def sites(path: Path):
    """Every call, the enclosing def, and the local name the buffer ends up under.

    Three binding forms occur: `out = alloc(...)`, `outs = [alloc(...) for _ in range(n)]` and
    `outs.append(alloc(...))`. The third has no assignment target, so the name comes off the
    append receiver; a grep-based audit silently drops it (it is triatt_qkv.py:417).
    """
    tree = ast.parse(path.read_text())
    parents = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parents[c] = n
    found = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and ast.unparse(n.func).endswith(
                "ttnn.allocate_tensor_on_device")):
            continue
        fn, node = None, n
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = node
                break
        target, node, form = None, n, "?"
        while node in parents:
            p_ = parents[node]
            if isinstance(p_, ast.Assign):
                target, form = ast.unparse(p_.targets[0]), "assign"
                break
            if isinstance(p_, ast.Call) and isinstance(p_.func, ast.Attribute) \
                    and p_.func.attr == "append":
                target, form = ast.unparse(p_.func.value), "append"
                break
            if isinstance(p_, (ast.ListComp, ast.Call, ast.Tuple, ast.List, ast.comprehension)):
                node = p_
                continue
            break
        found.append({"file": str(path.relative_to(ROOT)), "line": n.lineno,
                      "func": fn.name if fn else "<module>", "bound_to": target,
                      "bind_form": form, "_fn": fn})
    return found


def classify(site):
    """Where the bound name goes, and whether it comes back out.

    Names are matched as AST identifiers, not as substrings: `a` occurs inside the string literal
    `"p_a"` three lines below tenstorrent.py:5931 and a substring match calls that a use.
    An argument that is a list literal is reported by its index INSIDE the list, because
    `ttnn.generic_op([x, out], pd)` puts the destination at list position 1, not argument 0.
    """
    fn = site.pop("_fn")
    names = set()
    for t in (site["bound_to"] or "").replace("(", "").replace(")", "").split(","):
        t = t.strip().split("[")[0].split(".")[0]
        if t.isidentifier():
            names.add(t)
    uses, returned, aliases = [], False, []
    for _ in range(3):                     # follow aliasing one hop at a time
        grew = False
        for n in ast.walk(fn) if fn else []:
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) \
                    and ast.unparse(n.value.func) in ALIASING \
                    and _names(n.value) & names:
                for t in n.targets:
                    for nm in _names(t) - names:
                        names.add(nm)
                        aliases.append(f"{ast.unparse(n.value.func)} -> {nm}")
                        grew = True
        if not grew:
            break
    for n in ast.walk(fn) if fn else []:
        if isinstance(n, ast.Call):
            callee = ast.unparse(n.func)
            if callee in ALIASING:
                continue
            for k, arg in enumerate(n.args):
                if isinstance(arg, (ast.List, ast.Tuple)):
                    for j, el in enumerate(arg.elts):
                        if _names(el) & names:
                            uses.append((callee, f"io list position {j} of {len(arg.elts)}"))
                elif _names(arg) & names:
                    uses.append((callee, f"positional {k}"))
            for kw in n.keywords:
                if _names(kw.value) & names:
                    uses.append((callee, f"{kw.arg}="))
        if isinstance(n, ast.Return) and n.value is not None and _names(n.value) & names:
            returned = True
    dest = [(c, s) for c, s in uses
            if c in HELPERS or any(c.endswith(k.split(".")[-1]) for k in HELPERS)]
    site["aliased_as"] = aliases
    site["passed_to"] = sorted({f"{c} [{s}]" for c, s in dest})
    site["returned"] = returned
    site["n_writers"] = "1" if len(dest) <= 1 else f"{len(dest)} (partial, disjoint)"
    site["verdict"] = "WRITE" if dest else "UNCLASSIFIED"
    return site


def main():
    rows = []
    for p in sorted(PKG.rglob("*.py")):
        for s in sites(p):
            rows.append(classify(s))
    n_w = sum(r["verdict"] == "WRITE" for r in rows)
    print(f"{len(rows)} ttnn.allocate_tensor_on_device call sites in tt_bio/\n")
    print("%-22s %5s %-24s %-4s %-8s %s"
          % ("file", "line", "function", "ret", "writers", "destination slot"))
    for r in rows:
        print("%-22s %5d %-24s %-4s %-8s %s"
              % (Path(r["file"]).name, r["line"], r["func"][:24],
                 "yes" if r["returned"] else "NO", r["n_writers"],
                 "; ".join(r["passed_to"] + r["aliased_as"]) or "-- NONE --"))
    print(f"\nWRITE {n_w}/{len(rows)}   returned {sum(r['returned'] for r in rows)}/{len(rows)}")
    out = Path(__file__).resolve().parent / "prealloc_audit.json"
    out.write_text(json.dumps({"n_sites": len(rows), "n_write": n_w, "helpers": HELPERS,
                               "sites": rows}, indent=1))
    print("WROTE", out)
    return 0 if n_w == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())

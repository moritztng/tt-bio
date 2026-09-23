#!/usr/bin/env python3
"""Which ttnn verbs OpenFold3 calls, and which of them the tape can follow.

`ptx-fastpath` registered the tape on the VERBS rather than on Protenix, so the claim that
OF3 inherits differentiability is checkable without a device: take every ``ttnn.<verb>``
call site in OF3's own modules, intersect the names with ``tt_bio.taped_ttnn.VERBS``, and
name what is left over.

This is a static screen, not a proof. It cannot see a verb reached through an alias or a
``getattr``, it does not follow the shared helpers in ``tenstorrent.py`` that OF3 calls,
and it cannot tell whether a covered verb's tape entry is correct for the shapes OF3 hands
it. A verb missing here is a hard gap; a verb present here still has to survive a real
taped forward.

    python3 perf/of3t_perf/verb_census.py --out perf/of3t_perf/verb_census.json
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


#: Not every uncovered verb is a hole in the tape. ``from_torch`` mints a leaf and
#: ``to_torch`` ends the graph, which is what a tape boundary IS; ``zeros`` is a constant
#: with no gradient to carry. Anything else on this list is a real gap, and the run prints
#: those separately so a 90 %% coverage headline cannot hide a one-line hole.
KIND = {"from_torch": "leaf", "to_torch": "seam", "zeros": "constant",
        "ones": "constant", "full": "constant", "arange": "constant"}


def dotted(node):
    """``experimental.nlp_concat_heads`` out of an attribute chain, or ``""``."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    else:
        return ""
    return ".".join(reversed(parts))


def rebind_reach(path):
    """Whether ``tape()``'s module rebinding reaches this file's ``ttnn`` calls.

    ``_swap`` walks ``sys.modules`` and rebinds the module-global name ``ttnn``. A
    function-local ``import ttnn`` re-binds the REAL module into the function's own scope at
    call time, so those call sites run untaped no matter what the rebinding did. When such a
    site is handed a raw handle rather than an ``autograd.Tensor`` nothing raises, and the
    gradient stops there quietly -- which is the one failure mode the tape's "an op it cannot
    follow is loud" guarantee does not cover.
    """
    tree = ast.parse(path.read_text(), str(path))
    module_level = any(isinstance(n, (ast.Import, ast.ImportFrom))
                       and any(al.name.split(".")[0] == "ttnn" for al in n.names)
                       for n in tree.body)
    local = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(n):
                if isinstance(sub, ast.Import) and any(al.name == "ttnn" for al in sub.names):
                    local.append(sub.lineno)
    return module_level, sorted(set(local))


def scan(path):
    """Every ``ttnn.*`` call in one file: verb -> count, verb -> lines, plus aliased calls.

    The module-global name ``ttnn`` is what ``tape()`` rebinds, so a call spelled through a
    local alias is one the rebinding does not reach. Those are counted separately instead of
    being folded into the coverage number.
    """
    tree = ast.parse(path.read_text(), str(path))
    calls, aliased, lines = Counter(), Counter(), {}
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        q = dotted(n.func)
        if not q:
            continue
        head, _, rest = q.partition(".")
        if head == "ttnn" and rest:
            calls[rest] += 1
            lines.setdefault(rest, []).append(n.lineno)
        elif head in ("tnn", "_ttnn") and rest:
            aliased[rest] += 1
    return calls, aliased, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="tt_bio/openfold3*.py")
    ap.add_argument("--out", type=Path,
                    default=REPO / "perf" / "of3t_perf" / "verb_census.json")
    a = ap.parse_args()

    from tt_bio.taped_ttnn import VERBS
    taped = set(VERBS)

    files = sorted(REPO.glob(a.glob))
    per_file, total, aliased_total, lines = {}, Counter(), Counter(), {}
    blind = {}
    for f in files:
        c, al, ln = scan(f)
        mod_level, local = rebind_reach(f)
        per_file[f.name] = {"calls": sum(c.values()), "verbs": len(c),
                            "module_level_ttnn": mod_level, "local_import_lines": local}
        if sum(c.values()) and not mod_level:
            blind[f.name] = {"calls": sum(c.values()), "local_import_lines": local}
        total.update(c)
        aliased_total.update(al)
        for k, v in ln.items():
            lines.setdefault(k, []).extend(f"{f.name}:{ln}" for ln in v)

    covered = {v: n for v, n in total.items() if v in taped}
    missing = {v: n for v, n in total.items() if v not in taped}
    n_calls = sum(total.values())
    n_cov = sum(covered.values())

    out = {
        "glob": a.glob,
        "files": per_file,
        "taped_verbs_registered": sorted(taped),
        "ttnn_call_sites": n_calls,
        "ttnn_verbs_used": len(total),
        "covered_call_sites": n_cov,
        "covered_verbs": {v: n for v, n in sorted(covered.items(), key=lambda kv: -kv[1])},
        "uncovered_verbs": {v: {"calls": n, "kind": KIND.get(v, "gap"),
                                "sites": lines[v]}
                            for v, n in sorted(missing.items(), key=lambda kv: -kv[1])},
        "aliased_calls_not_scanned": dict(aliased_total),
        "rebind_blind_files": blind,
        "rebind_blind_call_sites": sum(v["calls"] for v in blind.values()),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"{len(files)} modules matching {a.glob}: {n_calls} ttnn call sites, "
          f"{len(total)} distinct verbs")
    print(f"tape registers {len(taped)} verbs; {n_cov}/{n_calls} call sites covered "
          f"({100.0 * n_cov / max(n_calls, 1):.1f} %)")
    print("UNCOVERED, by call count:")
    for v, n in sorted(missing.items(), key=lambda kv: -kv[1]):
        print(f"  {v:<16} {n:>4}  {KIND.get(v, 'gap'):<9}  {lines[v][0]}")
    for v in sorted(missing):
        if KIND.get(v, "gap") == "gap":
            for site in lines[v]:
                print(f"    GAP {v} {site}")
    if blind:
        n = sum(v["calls"] for v in blind.values())
        print(f"TAPE-BLIND: {n} call sites in {len(blind)} file(s) import ttnn inside a "
              f"function, so the module rebinding never reaches them:")
        for name, v in sorted(blind.items()):
            print(f"  {name:<32} {v['calls']:>4} calls, local import at line(s) "
                  f"{v['local_import_lines']}")
    print("WROTE", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

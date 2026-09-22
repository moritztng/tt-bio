#!/usr/bin/env python3
"""Which reductions in the tape carry `compute_kernel_config`, keyed by SYMBOL, not line number.

D55 is "the tape gives its precise kernel config to some reductions and withholds it from the ones
inside near-cancellations", and it has been re-located twice because every line number in it went
stale (pass 220, then again here). A defect located only by line number decays; this re-derives the
inventory from the AST on every run, so the answer cannot go stale and the next row does not spend a
pass finding the sites.

It also found that D55's own inventory was INCOMPLETE. The entry names four sites. Parsing the tree
finds **nine** unconfigured `ttnn.sum`/`ttnn.mean` calls, and the two it never listed are the most
interesting ones in the file -- see BACKWARD_MEAN below.

CPU only. Reads source. Opens no device, imports no ttnn.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REDUCERS = {"sum", "mean"}

# What each site is, so the table is readable without opening the file. Keyed by the enclosing
# function plus the expression, never by line.
WHAT = {
    "_sum_leading": "fan-in sum feeding a WEIGHT gradient -- configured, and the measurement that "
                    "motivated it is in D55: cosine 0.379 -> 0.999995",
    "layer_norm:mean": "the backward's recomputed E[x], whose result is subtracted from x three "
                       "lines later -- the near-cancellation this rule is built around",
    "layer_norm:var": "E[(x-mean)^2], the second pass -- configured",
    "layer_norm:dn_mean": "dx = (dnorm - mean(dnorm) - norm*mean(dnorm*norm))*rstd -- measured "
                          "INERT at pass 232, 2736/2736 bit-identical, LoFi break control moved "
                          "2733/2736",
    "softmax:inner": "inner = sum(g*y) -> x.grad = y*(g - inner) -- measured INERT at pass 222",
    "attention:inner": "inner = rowsum(dP*P) -> ds = P*(dP - inner), inside the chunked recompute "
                       "attention backward. UNMEASURED. The three matmuls around it all pass cfg",
    "attention:dbias": "bias-gradient accumulation over the query chunk. UNMEASURED",
}


def classify(fn: str, seg: str) -> str:
    s = " ".join(seg.split())
    if "_sum_leading" in fn:
        return "_sum_leading"
    if "dnorm, norm" in s or s.startswith("ttnn.mean(dnorm"):
        return "layer_norm:dn_mean"
    if "centered, centered" in s:
        return "layer_norm:var"
    if "ttnn.mean(xv" in s:
        return "layer_norm:mean"
    if "multiply(g, y)" in s:
        return "softmax:inner"
    if "multiply(dp, p)" in s:
        return "attention:inner"
    if "ttnn.sum(ds" in s:
        return "attention:dbias"
    return "other"


def enclosing(tree: ast.AST) -> dict[int, str]:
    """Map every line to the innermost def that contains it, so a site has a symbol."""
    out: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                out[ln] = node.name          # inner defs overwrite outer ones on the way down
    return out


def census(path: Path) -> list[dict]:
    src = path.read_text()
    tree = ast.parse(src)
    owner = enclosing(tree)
    rows = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in REDUCERS
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "ttnn"):
            continue
        seg = ast.get_source_segment(src, n) or ""
        kind = classify(owner.get(n.lineno, "?"), seg)
        rows.append({
            "line_now": n.lineno,                       # a convenience, NOT the identifier
            "enclosing_def": owner.get(n.lineno, "?"),
            "expr": " ".join(seg.split())[:120],
            "kind": kind,
            "configured": any(k.arg == "compute_kernel_config" for k in n.keywords if k.arg),
            "what": WHAT.get(kind, ""),
        })
    return sorted(rows, key=lambda r: r["line_now"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("tt_bio/autograd.py"))
    ap.add_argument("--report", type=Path, default=None)
    a = ap.parse_args()
    if not a.src.is_file():
        print(f"REFUSING: {a.src} does not exist -- run this from the composed tree")
        return 1

    rows = census(a.src)
    off = [r for r in rows if not r["configured"]]
    print(f"{len(rows)} ttnn.sum/mean calls in {a.src}; {len(off)} carry NO compute_kernel_config\n")
    for r in rows:
        print(f"  {'cfg' if r['configured'] else '---'}  {r['enclosing_def']:>14}  "
              f"{r['kind']:<22} {r['expr'][:74]}")

    print("\nUNCONFIGURED, grouped -- the measured ones first:")
    measured = {"layer_norm:dn_mean", "softmax:inner"}
    for k in sorted({r["kind"] for r in off}):
        n = sum(1 for r in off if r["kind"] == k)
        tag = "MEASURED INERT" if k in measured else "UNMEASURED"
        print(f"  {tag:<15} x{n}  {k}: {WHAT.get(k, '')}")

    # The finding this census exists for, stated where it cannot be missed.
    if any(r["kind"] == "layer_norm:mean" and not r["configured"] for r in rows) and \
       any(r["kind"] == "layer_norm:var" and r["configured"] for r in rows):
        print("\nASYMMETRY, and the rule's own comment argues against it: the LayerNorm backward "
              "\n  recomputes E[x] WITHOUT the precise config and E[(x-mean)^2] WITH it. The "
              "\n  comment three lines above says the two-pass form is used because "
              "E[x^2]-E[x]^2\n  'cancels catastrophically once the mean dominates the spread' -- "
              "so the author is\n  reasoning about exactly this cancellation, and the operand of "
              "the subtraction is the\n  unconfigured one while its consumer is configured. "
              "D55 never listed these two sites.")

    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"source": str(a.src), "n_calls": len(rows),
                   "n_unconfigured": len(off), "sites": rows},
                  open(a.report, "w"), indent=1, sort_keys=True)
        print(f"\nwrote {a.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""D55 deliverable 2: pull the four still-unconfigured reductions, with a LoFi break control.

D55's census (`perf/of3t_orchestrator/kcfgcensus/`) lists ten `ttnn.sum`/`ttnn.mean` calls in
the tape's backward rules that carry no `compute_kernel_config`. Six were measured INERT (four
`layer_norm:dn_mean` at pass 232, two `softmax:inner` at pass 222). Four were not. This measures
those four, together, on the diffusion scope.

THE CENSUS LINE NUMBERS ARE STALE AND ONE OF ITS ENTRIES NO LONGER EXISTS. The census cites
`attention:inner` as `ttnn.sum(ttnn.multiply(dp, p), dim=-1, keepdim=True)` at autograd.py:987.
D56's renorm repair deleted that expression: `triangle_attention`'s backward now calls the shared
`softmax_bw_inner`, whose reduction is the same line `taped_ttnn._v_softmax` reaches. So the
targets are located BY EXPRESSION AND ENCLOSING FUNCTION, never by line.

  T1  layer_norm.make.bw            mean = ttnn.mean(xv, dim=-1, keepdim=True)
  T2  _taped_layer_norm.make.bw     mean = ttnn.mean(xv, dim=-1, keepdim=True)
  T3  triangle_attention.make.bw    ttnn.sum(ds, dim=0, keepdim=True)      -- dbias
  T4  softmax_bw_inner              inner = ttnn.sum(ttnn.multiply(g, y), dim=dim, keepdim=True)

MEASUREMENT ONLY (AMENDMENT 1). Nothing in `tt_bio/` is edited. The injection is a wrapper on
`ttnn.mean`/`ttnn.sum` that fires only when the calling frame is one of the four target
statements, so it exists for the duration of this process and is reachable by no fold, no flag
and no default. All four targets are inside a backward rule (`def bw`, or `softmax_bw_inner`
whose only two callers are `autograd.py:1190` and `taped_ttnn.py:226`, both inside `bw`
closures), so none of them can execute in an inference fold at all.

Arms: `none` (baseline, twice for A/A), `pull` (precise_config on all four), `lofi` (the break
control -- LoFi, fp32_dest_acc off, on the same four; it must MOVE the reading or no flag
reached the kernel).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf", "of3t_diffusion"))

AUTOGRAD = "tt_bio/autograd.py"

# (enclosing qualname, verb, a substring that identifies the statement) -> target id
TARGETS = [
    ("layer_norm.make.bw", "mean", "ttnn.mean(xv, dim=-1"),
    ("_taped_layer_norm.make.bw", "mean", "ttnn.mean(xv, dim=-1"),
    ("triangle_attention.make.bw", "sum", "ttnn.sum(ds, dim=0"),
    ("softmax_bw_inner", "sum", "ttnn.sum(ttnn.multiply(g, y)"),
]


def _qualspans(path):
    src = pathlib.Path(path).read_text()
    tree = ast.parse(src)
    spans = []

    def walk(node, stack):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                ns = stack + [ch.name]
                spans.append((ch.lineno, ch.end_lineno, ".".join(ns)))
                walk(ch, ns)
            else:
                walk(ch, stack)

    walk(tree, [])
    return src, tree, spans


def resolve():
    """Locate the four targets in today's source. Returns {lineno: (tid, verb)} and a report."""
    src, tree, spans = _qualspans(AUTOGRAD)
    lines = src.splitlines()

    def qual_of(ln):
        enc = [s for s in spans if s[0] <= ln <= s[1]]
        enc.sort(key=lambda s: s[1] - s[0])
        return enc[0][2] if enc else "<module>"

    # every ttnn.sum / ttnn.mean Call node, with its full statement span
    calls = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if not (isinstance(f, ast.Attribute) and f.attr in ("sum", "mean")
                and isinstance(f.value, ast.Name) and f.value.id == "ttnn"):
            continue
        has_cfg = any(k.arg == "compute_kernel_config" for k in n.keywords)
        calls.append({"lineno": n.lineno, "end_lineno": n.end_lineno, "verb": f.attr,
                      "qual": qual_of(n.lineno), "configured": has_cfg,
                      "src": lines[n.lineno - 1].strip()})

    lmap, rep = {}, []
    for tid, (qual, verb, needle) in enumerate(TARGETS, 1):
        hit = [c for c in calls
               if c["qual"] == qual and c["verb"] == verb and needle in c["src"]]
        if len(hit) != 1:
            raise SystemExit(f"T{tid} {qual} {needle!r}: expected 1 match, got {len(hit)}")
        c = hit[0]
        if c["configured"]:
            raise SystemExit(f"T{tid} at :{c['lineno']} already carries a config -- census stale")
        for ln in range(c["lineno"], c["end_lineno"] + 1):
            lmap[ln] = (tid, verb)
        rep.append({"target": f"T{tid}", "qual": qual, "verb": verb,
                    "line_now": c["lineno"], "end_lineno": c["end_lineno"],
                    "src": c["src"], "configured_before": c["configured"]})
    census = {"n_ttnn_reductions": len(calls),
              "n_unconfigured": sum(1 for c in calls if not c["configured"]),
              "unconfigured": [{"line": c["lineno"], "qual": c["qual"], "src": c["src"]}
                               for c in calls if not c["configured"]]}
    return lmap, rep, census


FIRED = {}


def install(arm, lmap):
    import ttnn
    if arm == "none":
        return None
    if arm == "pull":
        from tt_bio.autograd import precise_config
        cfg = precise_config()
    elif arm == "lofi":
        cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True,
            fp32_dest_acc_en=False, packer_l1_acc=False)
    else:
        raise SystemExit(f"unknown arm {arm}")

    real_sum, real_mean = ttnn.sum, ttnn.mean
    af = os.path.abspath(AUTOGRAD)

    def wrap(real, verb):
        def w(*args, **kw):
            if "compute_kernel_config" not in kw:
                fr = sys._getframe(1)
                if fr.f_code.co_filename == af:
                    t = lmap.get(fr.f_lineno)
                    if t is not None and t[1] == verb:
                        FIRED[t[0]] = FIRED.get(t[0], 0) + 1
                        kw["compute_kernel_config"] = cfg
            return real(*args, **kw)
        return w

    ttnn.sum = wrap(real_sum, "sum")
    ttnn.mean = wrap(real_mean, "mean")
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("none", "pull", "lofi"))
    ap.add_argument("--fire-out", default="")
    a, rest = ap.parse_known_args()

    lmap, rep, census = resolve()
    print(json.dumps({"arm": a.arm, "targets": rep, "census_today": census}, indent=1),
          flush=True)

    install(a.arm, lmap)

    import device_gradient as DG
    sys.argv = [sys.argv[0]] + rest
    rc = DG.main()

    fired = {f"T{t}": FIRED.get(t, 0) for t in range(1, len(TARGETS) + 1)}
    print("FIRINGS " + json.dumps(fired), flush=True)
    if a.fire_out:
        pathlib.Path(a.fire_out).write_text(json.dumps(
            {"arm": a.arm, "fired": fired, "targets": rep, "census_today": census}, indent=1))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

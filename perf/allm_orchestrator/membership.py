#!/usr/bin/env python3
"""Which models execute which shared device ops, resolved from the tree.

The ALLM campaign's split depends on this: a lever lives in `tenstorrent.py`, so fixing its
gate helps exactly the models that construct the op it guards. The campaign brief inherited a
membership list that is wrong in three places, and each error sends a model to the wrong row.

Three traps this scan is written against, all of which produced a wrong answer first:

  - A docstring that names a class is not a use of it. `af2.py` mentions `PairformerLayer`
    only to say it deliberately does NOT subclass it.
  - The same NAME can be a different class. BoltzGen and rfd3 define their own torch
    `Transition` and `AttentionPairBias`; counting by name puts them in a core they are not in.
  - An ALIASED import hides the op. `boltzgen/model/models/boltz.py` does
    `PairformerModule as TTPairformerModule`, and a scan keyed on the local name reads BoltzGen
    as sharing nothing at all. That was this script's own first answer.

So a cell counts a construction only when the callee resolves, through that file's own imports,
to a name imported from `tt_bio.tenstorrent`, and every cell prints its sites. `Pairformer`
builds a list of `PairformerLayer` (tenstorrent.py:8903) and `PairformerModule` wraps
`Pairformer` (:11248), so all three names execute the same layer.

A zero means this scan found no construction. That is a lead to check by hand, not a proof of
absence -- it was wrong about BoltzGen once already.

    python3 perf/allm_orchestrator/membership.py            # table + sites
    python3 perf/allm_orchestrator/membership.py --json out.json
"""
import argparse
import ast
import json
import sys
from pathlib import Path

# A model is its own files. Shared modules (tenstorrent.py, reference.py) are deliberately
# absent: they are the code being shared, not a model that shares it.
MODELS = {
    "Boltz-2": ["tt_bio/boltz2.py"],
    "ESMFold2": ["tt_bio/esmfold2.py", "tt_bio/esmfold2_runtime.py"],
    "Protenix-v2": ["tt_bio/protenix.py", "tt_bio/protenix_template.py"],
    "OpenDDE": ["tt_bio/opendde.py"],
    "OpenFold3": sorted(str(p) for p in Path("tt_bio").glob("openfold3*.py")),
    "BoltzGen": sorted(str(p) for p in Path("tt_bio/boltzgen").rglob("*.py")),
    "RFdiffusion3": (sorted(str(p) for p in Path("tt_bio/rf3").rglob("*.py"))
                     + sorted(str(p) for p in Path("tt_bio/rfd3").rglob("*.py"))),
    "ESMC": ["tt_bio/esmc.py"],
    "Nesso-1": ["tt_bio/nesso1.py"],
    "AF2": ["tt_bio/af2.py", "tt_bio/af2_confidence.py"],
}

# The trunk ops the shipped lever families guard. PairformerLayer/Pairformer/PairformerModule
# are one unit reported three ways, because the call sites name it three ways.
PRIMS = ["PairformerLayer", "PairformerModule", "Pairformer", "TriangleMultiplication",
         "TriangleAttention", "Transition", "AttentionPairBias", "OuterProductMean"]
PAIRFORMER = {"PairformerLayer", "PairformerModule", "Pairformer"}


def scan(path):
    """-> {shared op: [line, ...]} for constructions in one file."""
    try:
        tree = ast.parse(Path(path).read_text(errors="replace"))
    except (SyntaxError, ValueError, UnicodeDecodeError):
        return {}
    # local name -> original name, so an alias still resolves to the op it is
    shared = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("tenstorrent"):
            for a in node.names:
                shared[a.asname or a.name] = a.name
    hits = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn, name = node.func, None
        if isinstance(fn, ast.Name):
            name = shared.get(fn.id)
        elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) \
                and fn.value.id == "tenstorrent":
            name = fn.attr                      # `tenstorrent.PairformerModule(...)`
        if name in PRIMS:
            hits.setdefault(name, []).append((node.lineno, _flags(node)))
    return hits


# Keyword arguments that decide WHICH route a Pairformer site takes, as opposed to what it
# computes. `fp32_softmax=True` is the one that matters most: tenstorrent.py:2189 turns it into
# `_gate_reject("site")`, so a site passing it never reaches the fused triangle attention, and
# rf3/remap.py:230 spells the two as mutually exclusive (`fp32_softmax=not on`).
ROUTING_KWARGS = ("fp32_softmax", "scale_pair_bias", "transpose_bias", "gated_move",
                  "accurate_softmax", "tri_att_sdpa_hifi", "tri_att_sdpa_ragged_pad",
                  "tri_att_accurate_softmax")


def _flags(node):
    """-> {kwarg: source text} for the routing kwargs this call site passes explicitly.

    Only what the site SPELLS. A kwarg absent here takes Pairformer's default, which for
    fp32_softmax is False -- and the absence is the point, so it must not be filled in.
    """
    out = {}
    for kw in node.keywords:
        if kw.arg in ROUTING_KWARGS:
            try:
                out[kw.arg] = ast.unparse(kw.value)
            except Exception:
                out[kw.arg] = "?"
        elif kw.arg is None:
            out["**"] = "**kwargs (unresolved -- read the site)"
    return out


# Names exported by tenstorrent.py that are plumbing rather than compute: importing them says
# nothing about whether a model shares the work the perf levers were landed in.
INFRA = {"Module", "TorchWrapper", "WeightScope", "Weights", "CORE_GRID_MAIN", "get_device",
         "set_fast_mode", "accurate_softmax_site", "triatt_sdpa_hifi_site", "sdpa_ragged_pad_site"}


def surface(files):
    """-> (compute names, helper names) this model imports from tenstorrent.py.

    Membership in the pairformer core is the wrong predictor on its own. What decides whether a
    model can receive work done in the shared tree is how much of the shared tree it executes at
    all, and a model that re-implements the stack in its own file shares nothing to fix.
    """
    names = set()
    for f in files:
        try:
            tree = ast.parse(Path(f).read_text(errors="replace"))
        except (SyntaxError, ValueError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and \
                    node.module.endswith("tenstorrent"):
                names |= {a.name for a in node.names}
            # `from . import tenstorrent` then `tenstorrent.PairformerModule(...)`. Counting only
            # ImportFrom read Boltz-2 as sharing NOTHING, which is the third time in this campaign
            # that a name-keyed scan was blind to a different access form -- after the alias and
            # the docstring. Blindness is always silent and always reads as absence.
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id == "tenstorrent":
                names.add(node.attr)
    compute = sorted(n for n in names if n not in INFRA and not n.startswith("_"))
    helper = sorted(n for n in names if n not in INFRA and n.startswith("_"))
    return compute, helper


def build():
    out = {}
    for model, files in MODELS.items():
        agg = {}
        for f in files:
            for op, entries in scan(f).items():
                agg.setdefault(op, []).extend(
                    {"site": f"{f}:{ln}", "flags": fl} for ln, fl in entries)
        out[model] = agg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="PATH", help="write the matrix as JSON too")
    args = ap.parse_args()
    if not Path("tt_bio").is_dir():
        print("run me from the repo root (no tt_bio/ here)", file=sys.stderr)
        return 2
    m = build()

    w = sys.stdout.write
    w(f"{'model':<14}" + "".join(f"{p[:13]:>15}" for p in PRIMS) + "   core?\n")
    for model, agg in m.items():
        core = "pairformer" if any(agg.get(p) for p in PAIRFORMER) else \
               ("trimul" if agg.get("TriangleMultiplication") else "-")
        w(f"{model:<14}" + "".join(f"{len(agg.get(p, [])):>15}" for p in PRIMS) + f"   {core}\n")
    w("\nsites, per entry\n")
    for model, agg in m.items():
        for op in PRIMS:
            for e in agg.get(op, []):
                w(f"  {model:<14} {op:<24} {e['site']}\n")

    w("\nrouting flags each Pairformer site SPELLS (absent = Pairformer's default)\n")
    for model, agg in m.items():
        for op in PAIRFORMER:
            for e in agg.get(op, []):
                fl = e["flags"]
                shown = ", ".join(f"{k}={v}" for k, v in sorted(fl.items())) or "(none: all defaults)"
                mark = "  <-- OFF the fused tri-att route" \
                    if fl.get("fp32_softmax") == "True" else ""
                w(f"  {model:<14} {e['site']:<46} {shown}{mark}\n")

    w("\nshared surface: what each model imports from tenstorrent.py\n")
    for model, files in MODELS.items():
        compute, helper = surface(files)
        w(f"  {model:<14} compute({len(compute)}): {', '.join(compute) or '-'}\n")
        if helper:
            w(f"  {'':<14} helpers:    {', '.join(helper)}\n")

    if args.json:
        out = {"sites": m, "surface": {k: dict(zip(("compute", "helpers"), surface(v)))
                                       for k, v in MODELS.items()}}
        Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        w(f"\nwrote {args.json}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

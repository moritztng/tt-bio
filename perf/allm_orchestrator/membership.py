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
            hits.setdefault(name, []).append(node.lineno)
    return hits


def build():
    out = {}
    for model, files in MODELS.items():
        agg = {}
        for f in files:
            for op, lines in scan(f).items():
                agg.setdefault(op, []).extend(f"{f}:{ln}" for ln in lines)
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
            if agg.get(op):
                w(f"  {model:<14} {op:<24} {' '.join(agg[op])}\n")

    if args.json:
        Path(args.json).write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
        w(f"\nwrote {args.json}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""35 of the engine's 90 matmul call sites pass no core_grid, and the hot path is among them.

`c10-fold-census` returned STOP on the 10.0 s target and then named one lead it could not resolve.
Measured on 27 of 33 matmul keys as an interleaved same-session variant, `core_grid=CORE_GRID_MAIN`
moves the fold's matmul class from **11.88 to 21.38 TFLOP/s**, 11.048 s to 6.1398 s -- **4.908 s /
6,626 Mcycles**. It was scrupulous about what that is not: "That 4.908 s is the gap between two of
MY replay configurations. It is NOT a fold saving and must not be entered on the ladder as one:
whether each Boltz-2 call site currently passes core_grid is a code question I did not resolve."

That is a code question, it needs no chip, and this answers it: which `ttnn.linear` and `ttnn.matmul`
call sites in `tt_bio/tenstorrent.py` pass `core_grid`, by AST rather than by grep, because these
calls span many lines and a grep on the call name cannot see which keywords belong to which call.
"""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parents[2] / "tt_bio" / "tenstorrent.py"
OPS = ("ttnn.linear", "ttnn.matmul")

# Classes and helpers on Boltz-2's default device path, from the census's own class table and the
# fold's call census. Named explicitly so the claim can be checked rather than taken on trust.
HOT = ("TriangleMultiplication", "TriangleAttention", "OuterProductMean", "DiffusionTransformer",
       "Diffusion", "AdaLN", "MiniTriangularUpdate", "_pair_proj_linear", "_narrow_proj_linear",
       "attn_value_matmul", "batched_matmul")
# The Fp32* classes are a separate precision path; whether the default fold reaches them is NOT
# resolved here and they are counted separately rather than folded into the hot-path total.
SEPARATE_PREFIX = "Fp32"
CENSUS = {"class_default_TFLOPs": 11.88, "class_best_TFLOPs": 21.38,
          "class_default_s": 11.048, "class_best_s": 6.1398,
          "difference_s": 4.908, "difference_Mcycles": 6626,
          "keys_varied": 27, "keys_total": 33,
          "caveat": "the census calls this a gap between two of ITS replay configurations, not a "
                    "fold saving, and forbids entering it on the ladder as one"}


def _qual(name):
    parts = []
    f = name.func
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return ".".join(reversed(parts))


def scan():
    src = ENGINE.read_text()
    tree = ast.parse(src)
    owner = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                owner.setdefault(ln, []).append(node.name)
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _qual(node) not in OPS:
            continue
        kws = node.keywords
        names = {k.arg for k in kws if k.arg}
        chain = owner.get(node.lineno) or ["<module>"]
        sites.append({
            "line": node.lineno, "op": _qual(node),
            "core_grid": "core_grid" in names,
            "kw_splat": any(k.arg is None for k in kws),
            "scope": " > ".join(chain[:2]), "owner": chain[-1], "chain": chain,
        })
    return sorted(sites, key=lambda s: s["line"])


def analyse():
    sites = scan()
    bare = [s for s in sites if not s["core_grid"] and not s["kw_splat"]]
    splat = [s for s in sites if not s["core_grid"] and s["kw_splat"]]
    hot = [s for s in bare if any(h in s["chain"] for h in HOT)]
    sep = [s for s in bare if any(c.startswith(SEPARATE_PREFIX) for c in s["chain"])]
    other = [s for s in bare if s not in hot and s not in sep]
    return {
        "scope": "CPU AST scan of the engine source. No device, no fold, no measurement, no lever "
                 "priced. Answers one code question the measuring row left open.",
        "engine": str(ENGINE),
        "census_lead": CENSUS,
        "totals": {"sites": len(sites),
                   "with_core_grid": sum(1 for s in sites if s["core_grid"]),
                   "no_core_grid_no_splat": len(bare),
                   "no_core_grid_but_splat": len(splat)},
        "bare_on_hot_path": sorted(hot, key=lambda s: s["line"]),
        "bare_on_separate_fp32_path": sorted(sep, key=lambda s: s["line"]),
        "bare_elsewhere": sorted(other, key=lambda s: s["line"]),
        "answer": (
            "%d of %d matmul call sites pass NO core_grid and have no **kw splat that could carry "
            "one, and %d of those are on Boltz-2's default device path -- TriangleMultiplication, "
            "TriangleAttention, OuterProductMean, DiffusionTransformer, Diffusion, AdaLN and the "
            "shared projection helpers. So the census's 4.908 s is NOT purely an artifact of its "
            "own replay configuration: the engine really does call these ops both ways."
            % (len(bare), len(sites), len(hot))),
        "what_this_does_NOT_establish": [
            "It does not price a lever. A site without an explicit core_grid gets ttnn's default, "
            "which may already be the right grid for that shape; the census measured a difference "
            "on ITS shapes, not on these sites.",
            "It does not map a call site to a census key. The launch-key census records class, "
            "output shape and K but not the call site, so which of these sites produces which "
            "measured key is unresolved.",
            "The Fp32* sites are counted separately because whether a default Boltz-2 fold reaches "
            "that precision path is not established here.",
            "This is exactly the shape of a one-size-tuning defect, which this fleet has on record "
            "as a standing class: a grid good for one shape can be wrong for another, so any change "
            "must be A/B'd PER SITE and never applied across the board.",
        ],
        "cheap_next_step": (
            "Read the five keys the census named -- 1x140x32x128 K=128 and K=256, 1x512x512x16 "
            "K=128, 1x16x512x256 K=64, and the top-FLOP 1x512x768 K=768 -- back to their call "
            "sites, A/B only those that are currently bare, and price the result as fold seconds in "
            "one interleaved session. That is a short session, not a table."),
    }


if __name__ == "__main__":
    r = analyse()
    (HERE / "core_grid_sites.json").write_text(json.dumps(r, indent=2) + "\n")
    t = r["totals"]
    print(f"{t['sites']} matmul call sites: {t['with_core_grid']} pass core_grid, "
          f"{t['no_core_grid_no_splat']} bare, {t['no_core_grid_but_splat']} rely on a **kw splat")
    print(f"bare on the hot path: {len(r['bare_on_hot_path'])}")
    for s in r["bare_on_hot_path"]:
        print(f"   L{s['line']:5d} {s['op']:12s} {s['scope']}")
    print(f"bare on the separate fp32 path: {len(r['bare_on_separate_fp32_path'])}; "
          f"bare elsewhere: {len(r['bare_elsewhere'])}")

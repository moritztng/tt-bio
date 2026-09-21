#!/usr/bin/env python3
"""Read `stagesplit.py` captures: check one, or put the two arms side by side.

`--check` is what the run chain calls after its smoke session, before it spends four more: a tape
that silently resolved no seam produces a plausible-looking JSON whose stages are all residual, so
the two seams that carry the question -- the trunk and the sampler -- are asserted present and
non-trivial rather than eyeballed.

With no `--check` it prints the old arm against the new arm, per stage, and the per-stage ratio
that the whole-fold ratio is made of.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

MUST = ("trunk", "diffusion_sample")


def load(p: Path):
    d = json.loads(Path(p).read_text())
    taped = [t.split(" -> ")[0] for t in d.get("taped", [])]
    untaped = [f["fold_s"] for f in d["folds"] if not f["taped"] and f["tag"] != "cold"]
    staged = [f for f in d["folds"] if f["taped"]]
    return d, taped, untaped, staged


def check(p: Path) -> int:
    d, taped, untaped, staged = load(p)
    bad = []
    for m in MUST:
        if m not in taped:
            bad.append(f"seam {m} never taped; taped={taped}")
    if not staged:
        bad.append("no taped fold in this capture")
    else:
        st = staged[-1]["stages"]
        for m in ("trunk", "diffusion"):
            if st.get(m, {}).get("share", 0) < 5.0:
                bad.append(f"stage {m} is {st.get(m)}, which is not a fold this tree ran")
        if st["residual"]["share"] > 40.0:
            bad.append(f"residual {st['residual']['share']} % -- the tape is mostly blind")
    for f in d["folds"]:
        c = f.get("clock", {})
        if c.get("aiclk_min", 0) < 1200:
            bad.append(f"fold {f['tag']} ran at aiclk_min {c.get('aiclk_min')}")
    if d["path_flags"].get("is_msa_compiled") or d["path_flags"].get("is_pairformer_compiled"):
        bad.append("a compiled trunk path bypasses the trunk seam")
    for b in bad:
        print("CHECK FAIL:", b)
    if bad:
        return 1
    print(f"CHECK OK: {p.name} taped={len(taped)} missing={len(d['missing'])} "
          f"stages={ {k: v['share'] for k, v in staged[-1]['stages'].items()} }")
    return 0


def arm(paths):
    """Median over sessions of each stage, plus the untaped fold wall those sessions ran."""
    walls, stage_s, clocks, plddt = [], {}, [], []
    for p in paths:
        d, _t, untaped, staged = load(p)
        walls += untaped
        for f in staged:
            for k, v in f["stages"].items():
                stage_s.setdefault(k, []).append(v["self_s"])
        for f in d["folds"]:
            c = f.get("clock", {})
            if c.get("aiclk_n"):
                clocks.append((c["aiclk_min"], c["aiclk_median"]))
        plddt += [f["plddt"] for f in d["folds"] if f.get("plddt")]
    return {"wall": statistics.median(walls) if walls else None,
            "walls": sorted(round(w, 4) for w in walls),
            "stages": {k: round(statistics.median(v), 4) for k, v in stage_s.items()},
            "aiclk_min": min(c[0] for c in clocks) if clocks else None,
            "plddt": plddt[0] if plddt else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=Path)
    ap.add_argument("--old", type=Path, nargs="*", default=[])
    ap.add_argument("--new", type=Path, nargs="*", default=[])
    a = ap.parse_args()
    if a.check:
        return check(a.check)
    o, n = arm(a.old), arm(a.new)
    print(f"untaped fold wall   old {o['wall']:.4f} s {o['walls']}")
    print(f"                    new {n['wall']:.4f} s {n['walls']}")
    print(f"whole fold ratio    {o['wall'] / n['wall']:.4f}x")
    print(f"aiclk min sampled   old {o['aiclk_min']}  new {n['aiclk_min']}")
    print(f"{'stage':<18}{'old s':>10}{'new s':>10}{'ratio':>9}{'seconds saved':>15}"
          f"{'% of saving':>13}")
    total = o["wall"] - n["wall"]
    for k in ("trunk", "diffusion", "confidence", "embed_and_heads", "residual"):
        if k not in o["stages"] or k not in n["stages"]:
            continue
        os_, ns = o["stages"][k], n["stages"][k]
        saved = os_ - ns
        print(f"{k:<18}{os_:>10.4f}{ns:>10.4f}{os_ / ns if ns else float('nan'):>9.4f}"
              f"{saved:>15.4f}{100.0 * saved / total:>13.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

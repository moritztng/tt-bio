"""Grade an abba.sh run: s/design per leg, AICLK beside it, ABBA pair ratios, designs.pt identity across arms.

    python perf/rfd4/p512/compare.py <outdir>/n512 [--floor 1200] > table.md; also writes <phase>/compare.json

A leg is one fair/run.py process; its s/design is warm wall over warm designs. A leg whose AICLK median (of its
batches' 1 Hz medians) is under --floor is discarded, not corrected, and so is every pair it is in. Pairs follow the
ABBA order the legs ran in: (A1, P1), (P2, A2), ... ; ratio = A s/design / P s/design, > 1 means P is faster.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch


def leg(d: Path):
    rows = [json.loads(l) for l in open(d / "runs.jsonl")]
    warm = [r for r in rows if r.get("kind") == "warm"]
    if not warm:
        return None
    clocks = [r["aiclk_mhz"] for r in warm if r.get("aiclk_mhz")]
    return {"leg": d.name, "s_per_design": sum(r["wall_s"] for r in warm) / sum(r["designs"] for r in warm),
            "aiclk_mhz": statistics.median(clocks) if clocks else None,
            "aiclk_min_mhz": min((r["aiclk_min_mhz"] for r in warm if r.get("aiclk_min_mhz")), default=None),
            "foreign_cores": max((r.get("foreign_cores_median") or 0) for r in warm),
            "foreign_holders": sorted({p for r in warm for p in r.get("foreign_device_holders") or []}),
            "width_achieved": min(r["width_achieved"] for r in warm),
            "batches": len(warm), "seeds": sorted(r["seed"] for r in warm)}


def same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if hasattr(a, "__dict__"):
        return type(a) is type(b) and same(vars(a), vars(b))
    return a == b


def identity(phase: Path):
    """Every designs.pt of one batch (width, phase, seed) against the first arm-A leg's, across all legs and both arms."""
    by_seed = {}
    for f in sorted(phase.glob("*/b*/*-s*/designs.pt")):
        by_seed.setdefault(f"{f.parent.parent.name}/{f.parent.name}", []).append(f)
    out = {}
    for seed, files in by_seed.items():
        ref = torch.load(files[0], weights_only=False)
        out[seed] = {"files": len(files), "legs": sorted({f.parts[-4] for f in files}),
                     "identical": all(same(ref, torch.load(f, weights_only=False)) for f in files[1:])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--floor", type=float, default=1200)
    a = ap.parse_args()
    phase = Path(a.phase)
    legs = [x for x in (leg(d) for d in sorted(phase.iterdir(), key=lambda p: p.stat().st_mtime)
                        if d.is_dir() and (d / "runs.jsonl").exists()) if x]
    for x in legs:
        x["discard"] = x["aiclk_mhz"] is None or x["aiclk_mhz"] < a.floor
    by = {x["leg"]: x for x in legs}
    order = sorted(by, key=lambda n: int(n[1:]))
    pairs = []
    for k in range(1, len(order) // 2 + 1):
        A, P = by.get(f"A{k}"), by.get(f"P{k}")
        if not (A and P):
            continue
        pairs.append({"pair": k, "ratio": A["s_per_design"] / P["s_per_design"],
                      "clock_gap_mhz": (P["aiclk_mhz"] or 0) - (A["aiclk_mhz"] or 0),
                      "graded": not (A["discard"] or P["discard"])})
    graded = [p["ratio"] for p in pairs if p["graded"]]
    ident = identity(phase)
    res = {"legs": legs, "pairs": pairs, "graded_pairs": len(graded),
           "median_pair": statistics.median(graded) if graded else None,
           "mean_s_per_design": {arm: statistics.mean(x["s_per_design"] for x in legs if x["leg"][0] == arm
                                                      and not x["discard"]) for arm in "AP"
                                 if any(x["leg"][0] == arm and not x["discard"] for x in legs)},
           "designs_identical": all(v["identical"] for v in ident.values()) if ident else None, "identity": ident}
    (phase / "compare.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"| leg | s/design | AICLK med (min) | foreign cores | holders | graded |\n|---|--:|---|--:|---|---|")
    for x in legs:
        print(f"| {x['leg']} | {x['s_per_design']:.3f} | {x['aiclk_mhz']} ({x['aiclk_min_mhz']}) | "
              f"{x['foreign_cores']:.2f} | {len(x['foreign_holders'])} | {'no' if x['discard'] else 'yes'} |")
    print("\npairs (A/P): " + ", ".join(f"{p['pair']}: {p['ratio']:.4f}{'' if p['graded'] else ' (discarded)'}"
                                        f" [{p['clock_gap_mhz']:+.0f} MHz]" for p in pairs))
    print(f"median graded pair: {res['median_pair']}  ({len(graded)} graded)")
    print(f"designs.pt identical across all legs: {res['designs_identical']} "
          f"({sum(v['files'] for v in ident.values())} files, {len(ident)} seeds)")


if __name__ == "__main__":
    main()

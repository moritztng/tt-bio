#!/usr/bin/env python3
"""Price a stack change on one box: seconds A/B, plus per-seed design agreement.

The perf page's PXDesign row is a within-row comparison, so every published GPU cell has to come
off one stack. Raising torch to reach Blackwell therefore moves the A100 and H200 cells too, and
this file is what decides whether that move is free. It reads the two sweeps' run reports, pools
the warm rounds of each stack, and pairs the written binder CIFs seed by seed.

The design comparison needs three scales side by side or the number means nothing:

  same stack, same seed, two processes   -> 0.0 A, or the sampler is not deterministic
  cross stack, same seed                 -> what the stack change did
  same stack, different seed             -> the fixture's own trajectory-to-trajectory spread

A cross-stack RMSD far below the inter-seed scale is a numerics perturbation. One comparable to it
means the two stacks landed on different designs, which is a real finding and not something to
publish through.

Direct RMSD is primary: both designs sit in the same target frame, so a rigid fit would hide a real
displacement. The Kabsch-fitted value is reported beside it.
"""
import argparse, glob, itertools, json, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cif_rmsd import coords, kabsch_rmsd, rmsd  # noqa: E402


def reports(root, label):
    out = []
    for f in sorted(glob.glob(os.path.join(root, "run_%s_rep*.json" % label))):
        out.append(json.load(open(f)))
    return out


def warm_cells(rep):
    return [r["gen_cell_s"] for r in rep.get("rounds", []) if not r.get("cold")]


def cifs_by_seed(rep):
    """seed -> written binder CIF, warm rounds only (the cold round repeats seed 0)."""
    out = {}
    for r in rep.get("rounds", []):
        if r.get("cold"):
            continue
        ds = (r.get("artifact") or {}).get("designs") or []
        if ds:
            out[r["seed"]] = ds[0]["path"]
    return out


def pool(root, label):
    reps = reports(root, label)
    cells, seeds = [], {}
    for i, rep in enumerate(reps):
        cells += warm_cells(rep)
        for s, p in cifs_by_seed(rep).items():
            seeds.setdefault(s, []).append(p)
    return reps, cells, seeds


def stats(xs):
    if not xs:
        return None
    m = st.median(xs)
    return {"median_s": round(m, 4), "n": len(xs), "min_s": round(min(xs), 4),
            "max_s": round(max(xs), 4), "spread_pct": round(100 * (max(xs) - min(xs)) / m, 3)}


def pair(a, b):
    xa, xb = coords(a), coords(b)
    return round(rmsd(xa, xb), 4), round(kabsch_rmsd(xa, xb), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="dir holding run_<label>_rep*.json")
    ap.add_argument("--pinned-label", default="laczc512_gen_pinned")
    ap.add_argument("--modern-label", default="laczc512_gen_modern")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rp, cp, sp = pool(a.results, a.pinned_label)
    rm, cm, sm = pool(a.results, a.modern_label)
    o = {"pinned": {"label": a.pinned_label, "seconds": stats(cp), "n_reports": len(rp)},
         "modern": {"label": a.modern_label, "seconds": stats(cm), "n_reports": len(rm)}}
    for k, reps in (("pinned", rp), ("modern", rm)):
        if reps:
            o[k]["stack"] = reps[0].get("env", {}).get("stack") or reps[0].get("stack")

    if o["pinned"]["seconds"] and o["modern"]["seconds"]:
        pm, mm = o["pinned"]["seconds"]["median_s"], o["modern"]["seconds"]["median_s"]
        o["seconds_delta_s"] = round(mm - pm, 4)
        o["seconds_delta_pct"] = round(100 * (mm - pm) / pm, 3)
        o["seconds_band_pct"] = round(max(2.0, o["pinned"]["seconds"]["spread_pct"],
                                          o["modern"]["seconds"]["spread_pct"]), 3)
        o["seconds_verdict"] = ("INSIDE-BAND" if abs(o["seconds_delta_pct"]) <= o["seconds_band_pct"]
                                else "OUTSIDE-BAND")

    # same stack, same seed, two processes: the determinism control
    same = []
    for tag, seeds in (("pinned", sp), ("modern", sm)):
        for s, paths in seeds.items():
            for x, y in itertools.combinations(paths, 2):
                d, f = pair(x, y)
                same.append({"stack": tag, "seed": s, "direct": d, "fitted": f})
    o["same_stack_same_seed"] = same
    o["determinism_ok"] = all(r["direct"] == 0.0 for r in same) if same else None

    # same stack, different seed: the fixture's own diversity
    inter = []
    for tag, seeds in (("pinned", sp), ("modern", sm)):
        ks = sorted(seeds)
        for s1, s2 in itertools.combinations(ks, 2):
            d, f = pair(seeds[s1][0], seeds[s2][0])
            inter.append({"stack": tag, "seeds": [s1, s2], "direct": d, "fitted": f})
    o["same_stack_diff_seed"] = inter
    o["inter_seed_mean_direct"] = round(st.mean([r["direct"] for r in inter]), 4) if inter else None

    # cross stack, same seed: what the stack change did
    cross = []
    for s in sorted(set(sp) & set(sm)):
        d, f = pair(sp[s][0], sm[s][0])
        cross.append({"seed": s, "direct": d, "fitted": f})
    o["cross_stack_same_seed"] = cross
    o["cross_stack_mean_direct"] = round(st.mean([r["direct"] for r in cross]), 4) if cross else None

    if o.get("inter_seed_mean_direct") and o.get("cross_stack_mean_direct") is not None:
        ratio = o["cross_stack_mean_direct"] / o["inter_seed_mean_direct"]
        o["cross_over_inter_ratio"] = round(ratio, 4)
        # 10 % pre-registered in state/pxdesign-perf-page-honest.md section 5, before the number existed
        o["design_verdict"] = "SAME-DESIGN" if ratio < 0.10 else "DESIGN-MOVED"
    json.dump(o, open(a.out, "w"), indent=1)
    print(json.dumps(o, indent=1))


if __name__ == "__main__":
    main()

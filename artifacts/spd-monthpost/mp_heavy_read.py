#!/usr/bin/env python3
"""mp_heavy_read.py ROOT PREFIX: c730 heavy-fold cells under ROOT/<PREFIX><old|new>-<model>-<mode>/ -> one JSON line per cell.

Runs on the host that holds the runs (python3 - ROOT PREFIX < this). fold_s = median of the warm reps from bench.jsonl,
AICLK = min of the per-rep in-fold medians and the overall min. Accuracy: top-ranked CIF (c730.cif) CA RMSD after Kabsch,
old vs new on the same seed (102..104, median), against the seed floor = new tree, seed 102 vs 103 vs 104 (median of pairs).
"""
import glob, json, os, statistics as st, sys
import numpy as np

ROOT, PRE = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""
SEEDS = (102, 103, 104)


def ca(path):
    cols, out = [], {}
    for ln in open(path):
        if ln.startswith("_atom_site."):
            cols.append(ln.split(".", 1)[1].strip())
        elif ln.startswith(("ATOM", "HETATM")) and cols:
            r = dict(zip(cols, ln.split()))
            if r.get("label_atom_id") == "CA":
                out[(r.get("label_asym_id"), r.get("label_seq_id"))] = [float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"])]
    return out


def rmsd(a, b):
    k = sorted(set(a) & set(b))
    P, Q = np.array([a[i] for i in k], float), np.array([b[i] for i in k], float)
    assert len(k) and np.isfinite(P).all() and np.isfinite(Q).all(), "non-finite or empty coordinates"
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ Q)
    S[-1] *= np.sign(np.linalg.det(U @ Vt))
    return float(np.sqrt(max((P ** 2).sum() + (Q ** 2).sum() - 2 * S.sum(), 0) / len(k)))


def top(d, s):
    p = f"{d}/struct_c730_s{s}/c730.cif"
    return ca(p) if os.path.exists(p) else None


cells = {}
for d in sorted(glob.glob(f"{ROOT}/{PRE}*-*")):
    n = os.path.basename(d)[len(PRE):]
    tag, rest = n.split("-", 1)
    model, mode = rest.rsplit("-", 1)
    reps, start, build = [], {}, {}
    try:
        for l in open(f"{d}/bench.jsonl"):
            r = json.loads(l)
            if r.get("ev") == "rep":
                reps.append(r)
            elif r.get("ev") == "start":
                start = r
            elif r.get("ev") == "build":
                build = r
    except OSError:
        pass
    warm = [r for r in reps if r.get("kind") == "warm" and r.get("fold_s") and not r.get("err")]
    clk = [c for r in warm for c in (r.get("aiclk") or {}).values()]
    cells[(tag, model, mode)] = d
    print(json.dumps(dict(cell=n, tag=tag, model=model, mode=mode, n=len(warm),
                          fold_s=round(st.median([r["fold_s"] for r in warm]), 2) if warm else None,
                          spread=round(max(r["fold_s"] for r in warm) - min(r["fold_s"] for r in warm), 2) if warm else None,
                          cold_s=next((round(r["fold_s"], 1) for r in reps if r.get("kind") != "warm" and r.get("fold_s")), None),
                          mhz_med=min((c["median"] for c in clk), default=None), mhz_min=min((c["min"] for c in clk), default=None),
                          finite=all(r.get("finite") for r in warm) if warm else None, digests=[r.get("digest") for r in warm],
                          samples=start.get("samples"), recycles=start.get("recycles"), fast_attr=build.get("fast"),
                          nlevers=len(build.get("levers") or []),
                          conf=round(st.median([r["metrics"]["confidence_score"] for r in warm if (r.get("metrics") or {}).get("confidence_score")]), 3) if warm else None,
                          iptm=round(st.median([r["metrics"]["iptm"] for r in warm if (r.get("metrics") or {}).get("iptm")]), 3) if warm else None, errs=[r.get("err") for r in reps if r.get("err")][:1])))

for (tag, model, mode), d in cells.items():
    if tag != "new" or ("old", model, mode) not in cells:
        continue
    o = cells[("old", model, mode)]
    try:
        same = [rmsd(top(o, s), top(d, s)) for s in SEEDS if top(o, s) and top(d, s)]
        floor = [rmsd(top(d, a), top(d, b)) for a, b in ((102, 103), (102, 104), (103, 104)) if top(d, a) and top(d, b)]
        ofloor = [rmsd(top(o, a), top(o, b)) for a, b in ((102, 103), (102, 104), (103, 104)) if top(o, a) and top(o, b)]
    except AssertionError as e:
        print(json.dumps(dict(acc=f"{model}-{mode}", err=str(e))))
        continue
    r2 = lambda v: round(st.median(v), 2) if v else None
    print(json.dumps(dict(acc=f"{model}-{mode}", old_vs_new=r2(same), seed_floor_new=r2(floor), seed_floor_old=r2(ofloor),
                          old_vs_new_all=[round(x, 2) for x in same])))

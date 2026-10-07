#!/usr/bin/env python3
"""Floor and fast-vs-exact from score.py's table, plus pose-to-pose deviation between top-ranked models.

    venv/bin/python analyze.py <data dir> <run root> preds.tsv > report.md

Top-ranked pose = the sample with the highest ranking_score within one seed (what a user keeps from one run).
FLOOR   = exact(s) vs exact(s'), s != s': what re-running exact with another seed moves.
FAST    = fast(s) vs exact(s), same seed: what switching mode moves. Same seed, so the comparison is paired.
Pose-to-pose uses the two predictions directly (one as the native): DockQ and binder RMSD after target fit.
Signed deltas fast - exact are averaged per complex over seeds, then a paired percentile bootstrap over complexes
(20,000 resamples) gives the pooled 95 % interval. A complex is flagged when any fast top-ranked DockQ leaves the
range spanned by its exact seeds, or its fast-vs-exact pose deviation exceeds the largest exact-vs-exact one.
"""
import csv, itertools, sys, tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score import score  # noqa: E402

M = ["dockq", "irmsd", "lrmsd", "tm_complex", "tm_binder", "plddt", "iptm", "ranking_score"]
data, root, tsv = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rows = list(csv.DictReader(open(tsv), delimiter="\t"))
top = {}
for r in rows:
    k = (r["pdb"], r["mode"], r["seed"])
    if k not in top or float(r["ranking_score"]) > float(top[k]["ranking_score"]):
        top[k] = r
pdbs = sorted({k[0] for k in top})
seeds = {m: sorted({k[2] for k in top if k[1] == m}) for m in ("exact", "fast")}
both = [s for s in seeds["exact"] if s in seeds["fast"]]
v = lambda p, m, s, c: float(top[(p, m, s)][c])


def cif(p, m, s):
    r = top[(p, m, s)]
    return next(root.glob(f"{m}_*/pred/{p}/seed_{s}/predictions/{p}_sample_{r['sample']}.cif"))


out = []
P = out.append
P(f"Complexes {len(pdbs)}; exact seeds {','.join(seeds['exact'])}; fast seeds {','.join(seeds['fast'])}; paired seeds {','.join(both)}.\n")

# Ground-truth metrics of the top-ranked pose: per complex, per mode, mean over seeds and the exact seed range.
P("## Top-ranked pose vs ground truth (mean over seeds; exact min-max in brackets)\n")
P("| PDB | DockQ exact | DockQ fast | iRMSD exact | iRMSD fast | LRMSD exact | LRMSD fast | TM exact | TM fast | ipTM exact | ipTM fast |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
flags, delta = [], defaultdict(list)
pose = {}
with tempfile.TemporaryDirectory() as tmp:
    n = 0
    for p in pdbs:
        ex = {c: [v(p, "exact", s, c) for s in seeds["exact"]] for c in M}
        fa = {c: [v(p, "fast", s, c) for s in seeds["fast"]] for c in M}
        cell = lambda c: (f"{np.mean(ex[c]):.3f} [{min(ex[c]):.2f}-{max(ex[c]):.2f}]", f"{np.mean(fa[c]):.3f}")
        P(f"| {p} | " + " | ".join(" | ".join(cell(c)) for c in ("dockq", "irmsd", "lrmsd", "tm_complex", "iptm")) + " |")
        for c in M:
            delta[c].append(np.mean([v(p, "fast", s, c) - v(p, "exact", s, c) for s in both]))
        # pose-to-pose: exact vs exact across seeds (floor), fast vs exact same seed
        fl = []
        for a, b in itertools.combinations(seeds["exact"], 2):
            n += 1
            fl.append(score(cif(p, "exact", a), cif(p, "exact", b), tmp, f"p{n}"))
        fx = []
        for s in both:
            n += 1
            fx.append(score(cif(p, "fast", s), cif(p, "exact", s), tmp, f"p{n}"))
        pose[p] = (fl, fx)
        why = []
        lo, hi = min(ex["dockq"]), max(ex["dockq"])
        if any(not lo <= d <= hi for d in fa["dockq"]):
            why.append(f"fast DockQ {','.join(f'{d:.2f}' for d in fa['dockq'])} outside exact range {lo:.2f}-{hi:.2f}")
        if fx and fl and max(x["lrmsd"] for x in fx) > max(x["lrmsd"] for x in fl):
            why.append(f"fast-vs-exact binder deviation {max(x['lrmsd'] for x in fx):.2f} A > largest exact-vs-exact {max(x['lrmsd'] for x in fl):.2f} A")
        if why:
            flags.append((p, why))

P("\n## Success rate of the top-ranked pose (DockQ >= 0.23 / >= 0.49 / >= 0.80), all complex x seed runs\n")
for m in ("exact", "fast"):
    d = [v(p, m, s, "dockq") for p in pdbs for s in seeds[m]]
    P(f"* {m}: {np.mean(np.array(d) >= .23):.1%} acceptable, {np.mean(np.array(d) >= .49):.1%} medium, {np.mean(np.array(d) >= .80):.1%} high (n={len(d)}); mean DockQ {np.mean(d):.3f}")

P("\n## FLOOR vs FAST, ground-truth metrics of the top-ranked pose (absolute difference, median / mean over complex x pair)\n")
P("| metric | FLOOR exact(s) vs exact(s') | FAST fast(s) vs exact(s) | paired mean fast - exact [95 % CI] |")
P("|---|---|---|---|")
rng = np.random.default_rng(0)
for c in M:
    fl = [abs(v(p, "exact", a, c) - v(p, "exact", b, c)) for p in pdbs for a, b in itertools.combinations(seeds["exact"], 2)]
    fx = [abs(v(p, "fast", s, c) - v(p, "exact", s, c)) for p in pdbs for s in both]
    d = np.array(delta[c])
    boot = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
    P(f"| {c} | {np.median(fl):.3f} / {np.mean(fl):.3f} | {np.median(fx):.3f} / {np.mean(fx):.3f} | "
      f"{d.mean():+.4f} [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}] |")

P("\n## Pose-to-pose deviation of top-ranked models (median / max per complex)\n")
P("| PDB | exact vs exact: DockQ | binder RMSD A | TM | fast vs exact same seed: DockQ | binder RMSD A | TM |")
P("|---|---|---|---|---|---|---|")
for p in pdbs:
    fl, fx = pose[p]
    f = lambda xs, c, agg: agg([x[c] for x in xs]) if xs else float("nan")
    P(f"| {p} | {f(fl,'dockq',np.median):.3f} / min {f(fl,'dockq',min):.3f} | {f(fl,'lrmsd',np.median):.2f} / {f(fl,'lrmsd',max):.2f} | "
      f"{f(fl,'tm_complex',np.median):.3f} | {f(fx,'dockq',np.median):.3f} / min {f(fx,'dockq',min):.3f} | "
      f"{f(fx,'lrmsd',np.median):.2f} / {f(fx,'lrmsd',max):.2f} | {f(fx,'tm_complex',np.median):.3f} |")
allfl = [x["lrmsd"] for p in pdbs for x in pose[p][0]]
allfx = [x["lrmsd"] for p in pdbs for x in pose[p][1]]
P(f"\nPooled binder RMSD between top poses: exact vs exact median {np.median(allfl):.2f} A (n={len(allfl)}), "
  f"fast vs exact same seed median {np.median(allfx):.2f} A (n={len(allfx)}).")

P("\n## Flags\n")
P("\n".join(f"* {p}: " + "; ".join(w) for p, w in flags) if flags else "* none")
print("\n".join(out))

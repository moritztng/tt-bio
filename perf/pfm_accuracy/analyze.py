#!/usr/bin/env python3
"""Floor and fast-vs-exact from score.py's table, plus pose-to-pose deviation between top-ranked models.

    venv/bin/python analyze.py <data dir> <run root> preds.tsv [BASE TEST] > report.md

BASE/TEST default to exact/fast (the kit modes); any two `mode` values of preds.tsv work, e.g. exact ttexact.

Top-ranked pose = the sample with the highest ranking_score within one seed (what a user keeps from one run).
FLOOR   = exact(s) vs exact(s'), s != s': what re-running exact with another seed moves.
FAST    = fast(s) vs exact(s), same seed: what switching mode moves. Same seed, so the comparison is paired.
Pose-to-pose uses the two predictions directly (one as the native): DockQ and binder RMSD after target fit, on the
residues the deposited structure resolves.
Signed deltas fast - exact are averaged per complex over seeds, then a paired percentile bootstrap over complexes
(20,000 resamples) gives the pooled 95 % interval. A complex is flagged when any fast top-ranked DockQ leaves the
range spanned by its exact seeds by more than 0.05, or its fast-vs-exact pose deviation exceeds the largest exact-vs-exact one.
"""
import csv, itertools, sys, tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score import chains, score  # noqa: E402

TOL = 0.05  # DockQ slack around the exact seed range before a fast pose counts as outside it
M = ["dockq", "irmsd", "lrmsd", "tm_complex", "tm_binder", "plddt", "iptm", "ranking_score"]
data, root, tsv = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
BASE, TEST = sys.argv[4:6] if len(sys.argv) > 5 else ("exact", "fast")
names = {l.split("\t")[0]: l.split("\t")[1:3] for l in (Path(__file__).parent / "set.tsv").read_text().splitlines()[1:]}
rows = list(csv.DictReader(open(tsv), delimiter="\t"))
top = {}
for r in rows:
    k = (r["pdb"], r["mode"], r["seed"])
    if k not in top or float(r["ranking_score"]) > float(top[k]["ranking_score"]):
        top[k] = r
pdbs = sorted({k[0] for k in top})
seeds = {m: sorted({k[2] for k in top if k[1] == m}) for m in (BASE, TEST)}
both = [s for s in seeds[BASE] if s in seeds[TEST]]
v = lambda p, m, s, c: float(top[(p, m, s)][c])


def cif(p, m, s):
    r = top[(p, m, s)]
    return next(root.glob(f"{m}_*/pred/{p}/seed_{s}/predictions/{p}_sample_{r['sample']}.cif"))


out = []
P = out.append
P(f"Complexes {len(pdbs)}; {BASE} seeds {','.join(seeds[BASE])}; {TEST} seeds {','.join(seeds[TEST])}; paired seeds {','.join(both)}.\n")

# Ground-truth metrics of the top-ranked pose: per complex, per mode, mean over seeds and the exact seed range.
P(f"## Top-ranked pose vs ground truth (mean over seeds; {BASE} min-max in brackets)\n")
P(f"| PDB | DockQ {BASE} | DockQ {TEST} | iRMSD {BASE} | iRMSD {TEST} | LRMSD {BASE} | LRMSD {TEST} | TM {BASE} | TM {TEST} | ipTM {BASE} | ipTM {TEST} |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
flags, delta = [], defaultdict(list)
pose = {}
with tempfile.TemporaryDirectory() as tmp:
    n = 0
    for p in pdbs:
        ex = {c: [v(p, BASE, s, c) for s in seeds[BASE]] for c in M}
        fa = {c: [v(p, TEST, s, c) for s in seeds[TEST]] for c in M}
        cell = lambda c: (f"{np.mean(ex[c]):.3f} [{min(ex[c]):.2f}-{max(ex[c]):.2f}]", f"{np.mean(fa[c]):.3f}")
        P(f"| {p} | " + " | ".join(" | ".join(cell(c)) for c in ("dockq", "irmsd", "lrmsd", "tm_complex", "iptm")) + " |")
        for c in M:
            delta[c].append(np.mean([v(p, TEST, s, c) - v(p, BASE, s, c) for s in both]))
        # pose-to-pose: exact vs exact across seeds (floor), fast vs exact same seed, on deposited residues
        only = chains(data / "ref" / f"{p}.cif", names[p])
        fl = []
        for a, b in itertools.combinations(seeds[BASE], 2):
            n += 1
            fl.append(score(cif(p, BASE, a), cif(p, BASE, b), None, tmp, f"p{n}", only))
        fx = []
        for s in both:
            n += 1
            fx.append(score(cif(p, TEST, s), cif(p, BASE, s), None, tmp, f"p{n}", only))
        pose[p] = (fl, fx)
        why = []
        lo, hi = min(ex["dockq"]), max(ex["dockq"])
        if any(not lo - TOL <= d <= hi + TOL for d in fa["dockq"]):
            why.append(f"{TEST} DockQ {','.join(f'{d:.2f}' for d in fa['dockq'])} outside {BASE} range {lo:.2f}-{hi:.2f}")
        if fx and fl and max(x["lrmsd"] for x in fx) > max(x["lrmsd"] for x in fl):
            why.append(f"{TEST}-vs-{BASE} binder deviation {max(x['lrmsd'] for x in fx):.2f} A > largest {BASE}-vs-{BASE} {max(x['lrmsd'] for x in fl):.2f} A")
        if why:
            flags.append((p, why))

P("\n## Success rate of the top-ranked pose (DockQ >= 0.23 / >= 0.49 / >= 0.80), all complex x seed runs\n")
for m in (BASE, TEST):
    d = [v(p, m, s, "dockq") for p in pdbs for s in seeds[m]]
    P(f"* {m}: {np.mean(np.array(d) >= .23):.1%} acceptable, {np.mean(np.array(d) >= .49):.1%} medium, {np.mean(np.array(d) >= .80):.1%} high (n={len(d)}); mean DockQ {np.mean(d):.3f}")

# Docking success is bimodal per seed (the 5 samples share one trunk), so the seed is the unit. Discordance =
# one run of a pair acceptable (DockQ >= 0.23) and the other not: across seeds for the floor, same seed for fast.
P("\n## Acceptable top-ranked pose per complex (seeds with DockQ >= 0.23) and discordance against the floor\n")
P(f"| PDB | {BASE} | {TEST} | {BASE} vs {BASE} discordant pairs | {TEST} vs {BASE} same seed discordant |")
P("|---|---|---|---|---|")
ok = lambda p, m, s: v(p, m, s, "dockq") >= .23
dfl = dfx = nfl = nfx = 0
b = c = 0
for p in pdbs:
    a_ = sum(ok(p, a, x) != ok(p, a, y) for a in [BASE] for x, y in itertools.combinations(seeds[BASE], 2))
    f_ = sum(ok(p, TEST, x) != ok(p, BASE, x) for x in both)
    b += sum(ok(p, BASE, x) and not ok(p, TEST, x) for x in both)
    c += sum(ok(p, TEST, x) and not ok(p, BASE, x) for x in both)
    k = len(seeds[BASE]) * (len(seeds[BASE]) - 1) // 2
    dfl, nfl, dfx, nfx = dfl + a_, nfl + k, dfx + f_, nfx + len(both)
    P(f"| {p} | {sum(ok(p, BASE, x) for x in seeds[BASE])}/{len(seeds[BASE])} | {sum(ok(p, TEST, x) for x in seeds[TEST])}/{len(seeds[TEST])} | {a_}/{k} | {f_}/{len(both)} |")
from math import comb
pm = min(1.0, 2 * sum(comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c)) if b + c else 1.0
P(f"\nDiscordance: {BASE} vs {BASE} {dfl}/{nfl} = {dfl/nfl:.1%}, {TEST} vs {BASE} same seed {dfx}/{nfx} = {dfx/nfx:.1%}. "
  f"Same-seed flips: {BASE} ok -> {TEST} not {b}, {TEST} ok -> {BASE} not {c}; exact McNemar p = {pm:.2f}.")

P("\n## FLOOR vs FAST, ground-truth metrics of the top-ranked pose (absolute difference, median / mean over complex x pair)\n")
P(f"| metric | FLOOR {BASE}(s) vs {BASE}(s') | FAST {TEST}(s) vs {BASE}(s) | paired mean {TEST} - {BASE} [95 % CI] |")
P("|---|---|---|---|")
rng = np.random.default_rng(0)
for c in M:
    fl = [abs(v(p, BASE, a, c) - v(p, BASE, b, c)) for p in pdbs for a, b in itertools.combinations(seeds[BASE], 2)]
    fx = [abs(v(p, TEST, s, c) - v(p, BASE, s, c)) for p in pdbs for s in both]
    d = np.array(delta[c])
    boot = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
    P(f"| {c} | {np.median(fl):.3f} / {np.mean(fl):.3f} | {np.median(fx):.3f} / {np.mean(fx):.3f} | "
      f"{d.mean():+.4f} [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}] |")

P("\n## Pose-to-pose deviation of top-ranked models (median / max per complex)\n")
P(f"| PDB | {BASE} vs {BASE}: DockQ | binder RMSD A | TM | {TEST} vs {BASE} same seed: DockQ | binder RMSD A | TM |")
P("|---|---|---|---|---|---|---|")
for p in pdbs:
    fl, fx = pose[p]
    f = lambda xs, c, agg: agg([x[c] for x in xs]) if xs else float("nan")
    P(f"| {p} | {f(fl,'dockq',np.median):.3f} / min {f(fl,'dockq',min):.3f} | {f(fl,'lrmsd',np.median):.2f} / {f(fl,'lrmsd',max):.2f} | "
      f"{f(fl,'tm_complex',np.median):.3f} | {f(fx,'dockq',np.median):.3f} / min {f(fx,'dockq',min):.3f} | "
      f"{f(fx,'lrmsd',np.median):.2f} / {f(fx,'lrmsd',max):.2f} | {f(fx,'tm_complex',np.median):.3f} |")
allfl = [x["lrmsd"] for p in pdbs for x in pose[p][0]]
allfx = [x["lrmsd"] for p in pdbs for x in pose[p][1]]
P(f"\nPooled binder RMSD between top poses: {BASE} vs {BASE} median {np.median(allfl):.2f} A (n={len(allfl)}), "
  f"{TEST} vs {BASE} same seed median {np.median(allfx):.2f} A (n={len(allfx)}).")

P("\n## Flags\n")
P("\n".join(f"* {p}: " + "; ".join(w) for p, w in flags) if flags else "* none")
print("\n".join(out))

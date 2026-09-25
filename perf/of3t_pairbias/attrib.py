"""Which half of the split moved the served structure: the trunk or the confidence head?

`tonly` is this branch with the confidence head's call reverted to exactly what `wk/of3t`
makes (`scale_pair_bias=True`, no tri override). If `tonly` tracks `fix`, the ranking
change belongs to the trunk; if it tracks `ship`, it belongs to the confidence head.
"""
import json
import os
import statistics

import gemmi

GT = "examples/ground_truth_structures/ubiquitin.pdb"
ROOTS = {"ship": "/tmp/of3t/of3t-pairbias/fold", "fix": "/tmp/of3t/of3t-pairbias/fold",
         "tonly": "/tmp/of3t/of3t-pairbias/fold2"}
OUT = "perf/of3t_pairbias/attribution.json"


def ca(p):
    st = gemmi.read_structure(p)
    st.remove_alternative_conformations()
    return [r.find_atom("CA", "*").pos for c in st[0] for r in c
            if r.find_atom("CA", "*") is not None]


def rmsd(a, b):
    n = min(len(a), len(b))
    return gemmi.superpose_positions(a[:n], b[:n]).rmsd


gt = ca(GT)
rows, rep = {}, {}
for arm, root in ROOTS.items():
    for s in range(1, 10):
        d = os.path.join(root, f"{arm}_s{s}", "openfold3_results_ubq")
        sd = os.path.join(d, "structures")
        if not os.path.isdir(sd):
            continue
        fs = [os.path.join(sd, "ubq.cif")] + [os.path.join(sd, f"ubq_model_{i}.cif")
                                              for i in range(1, 5)]
        ms = [ca(f) for f in fs if os.path.exists(f)]
        c = json.load(open(os.path.join(d, "results.json")))[0]
        rows[(arm, s)] = ([rmsd(m, gt) for m in ms], ms, c)

seeds = sorted({s for (_, s) in rows})
print(f"{'seed':>4}  " + "".join(f"{a:>26}" for a in ROOTS))
for s in seeds:
    line = f"{s:>4}  "
    for a in ROOTS:
        if (a, s) in rows:
            v, _, c = rows[(a, s)]
            line += f"  rank0 {v[0]:5.3f} best {min(v):5.3f} pl {c['plddt']:.3f}"
        else:
            line += " " * 26
    print(line)

for a in ROOTS:
    vs = [rows[(a, s)] for s in seeds if (a, s) in rows]
    if not vs:
        continue
    r0 = [v[0][0] for v in vs]
    bs = [min(v[0]) for v in vs]
    pl = [v[2]["plddt"] for v in vs]
    rep[a] = dict(n=len(vs), rank0_mean=statistics.fmean(r0), best_mean=statistics.fmean(bs),
                  plddt_mean=statistics.fmean(pl), rank0=r0)
    print(f"  -> {a:6s} n={len(vs)}  rank0 mean {statistics.fmean(r0):.3f} A  "
          f"best-of-5 mean {statistics.fmean(bs):.3f} A  plddt {statistics.fmean(pl):.4f}")

print("\nstructure distance between arms, matched seed, rank 0")
for x, y in (("ship", "tonly"), ("fix", "tonly"), ("ship", "fix")):
    ds = [rmsd(rows[(x, s)][1][0], rows[(y, s)][1][0]) for s in seeds
          if (x, s) in rows and (y, s) in rows]
    if ds:
        rep[f"{x}_vs_{y}"] = dict(n=len(ds), mean=statistics.fmean(ds), values=ds)
        print(f"  {x:5s} vs {y:5s}  n={len(ds)} mean {statistics.fmean(ds):5.3f} A   "
              + " ".join(f"{d:.3f}" for d in ds))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(rep, open(OUT, "w"), indent=1)
print("\nwrote", OUT)

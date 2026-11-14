"""Grade a test arm against a base arm on the ligand set (make_set.py), by ligand pose.

    python perf/opendde_lig/grade_lig.py RUN --base old --test new [--data ~/odfp-lig-data] [--out grade.md]

RUN/<arm>/<name>/bench.jsonl are bench.py records; each names the struct_dir holding the rank-0 sample <name>.cif.
Pocket = reference residues with a CA within 10 A of a ligand heavy atom. A pose is fitted on the pocket CAs and the
ligand heavy atoms are compared by (residue order, CCD atom name), so no symmetry correction: both arms carry the
same bias. Per complex and seed:
  LRMSD    ligand RMSD vs the deposited pose (success: < 2 A, the PoseBusters bar)
  PAIRED   test LRMSD - base LRMSD, same seed
  DEV      test pose vs base pose, same seed (ligand RMSD after a pocket fit), in A
  FLOOR    base seed s vs base seed s' (every pair), the same deviation: what re-seeding the base moves
Verdict: every fold present (>= 4 paired seeds per complex); median DEV <= max(0.60 A, median FLOOR); success
count not lower than base by more than one fold; mean PAIRED's 95 % bootstrap interval over complexes reaches 0
or lies below it.
"""
import argparse, itertools, json, statistics
from pathlib import Path

import gemmi
import numpy as np

KILL_BAR, POCKET = 0.60, 10.0


def load(cif):
    st = gemmi.read_structure(str(cif))
    st.setup_entities()
    ca, lig, k = {}, {}, 0
    plddt = statistics.mean(a.b_iso for ch in st[0] for r in ch for a in r) / 100  # writers put pLDDT in B_iso
    for ch in st[0]:
        pol = ch.get_polymer()
        if len(pol):
            for r in pol:
                a = r.find_atom("CA", "*")
                if a and r.label_seq is not None:
                    ca[(k, r.label_seq)] = np.array(a.pos.tolist())
            k += 1
        n = 0
        for r in ch:
            if r.entity_type in (gemmi.EntityType.NonPolymer, gemmi.EntityType.Branched):
                for a in r:
                    if a.element.name != "H":
                        lig[(n, a.name)] = np.array(a.pos.tolist())
                n += 1
    return ca, lig, plddt


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return R, qc - pc @ R


def lig_dev(a, b, pocket):
    """Ligand RMSD of pose a against pose b after fitting a's pocket CAs onto b's."""
    keys = [k for k in pocket if k in a[0] and k in b[0]]
    R, t = kabsch(np.array([a[0][k] for k in keys]), np.array([b[0][k] for k in keys]))
    atoms = [k for k in b[1] if k in a[1]]
    if len(atoms) < 0.9 * len(b[1]):
        raise ValueError(f"only {len(atoms)}/{len(b[1])} ligand atoms match by name")
    A, B = np.array([a[1][k] for k in atoms]), np.array([b[1][k] for k in atoms])
    return float(np.sqrt((((A @ R + t) - B) ** 2).sum(-1).mean()))


def folds(run, arm, name):
    out = {}
    f = run / arm / name / "bench.jsonl"
    for line in f.read_text().splitlines() if f.exists() else []:
        r = json.loads(line)
        if r.get("fold_s") is not None and r.get("struct_dir"):
            cif = Path(r["struct_dir"]) / f"{name}.cif"
            cif = cif if cif.exists() else run / arm / name / cif.parent.name / cif.name  # rsynced run
            out[r["seed"]] = load(cif)
    return out


def boot_ci(x, n=20000, seed=0):
    x = np.asarray(x)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--base", default="old")
ap.add_argument("--test", default="new")
ap.add_argument("--data", type=Path, default=Path("~/odfp-lig-data").expanduser())
ap.add_argument("--out", type=Path)
a = ap.parse_args()

names = [l.split("\t")[0] for l in (a.data / "set.tsv").read_text().splitlines()[1:]]
rows, paired, devs, floors, succ, insufficient = [], [], [], [], {a.base: 0, a.test: 0}, []
for name in names:
    ref = load(a.data / "ref" / f"{name}.cif")
    L = np.array(list(ref[1].values()))
    pocket = [k for k, x in ref[0].items() if np.linalg.norm(L - x, axis=1).min() < POCKET]
    base, test = folds(a.run, a.base, name), folds(a.run, a.test, name)
    seeds = sorted(set(base) & set(test))
    if len(seeds) < 4:
        insufficient.append(f"{name} ({len(seeds)} paired seeds)")
        if not seeds:
            continue
    lb = {s: lig_dev(base[s], ref, pocket) for s in seeds}
    lt = {s: lig_dev(test[s], ref, pocket) for s in seeds}
    d = [lig_dev(test[s], base[s], pocket) for s in seeds]
    fl = [lig_dev(base[s], base[u], pocket) for s, u in itertools.combinations(sorted(base), 2)]
    p = [lt[s] - lb[s] for s in seeds]
    succ[a.base] += sum(v < 2 for v in lb.values()); succ[a.test] += sum(v < 2 for v in lt.values())
    paired.append(statistics.mean(p)); devs += d; floors += fl
    pl = [test[s][2] - base[s][2] for s in seeds]
    rows.append(f"| {name} | {len(seeds)} | {statistics.median(lb.values()):.2f} | {statistics.median(lt.values()):.2f} "
                f"| {statistics.mean(p):+.2f} | {statistics.median(d):.2f} | {statistics.median(fl):.2f} "
                f"| {statistics.mean(pl):+.4f} |")

md = [f"Ligand set, {a.test} vs {a.base} (base seed floor beside every deviation; LRMSD = ligand RMSD after a "
      f"{POCKET:.0f} A pocket fit, A)", "",
      f"| complex | seeds | LRMSD {a.base} | LRMSD {a.test} | PAIRED mean | DEV median | FLOOR median | pLDDT delta |",
      "|---|---|---|---|---|---|---|---|", *rows, ""]
verdict = "INSUFFICIENT: " + ", ".join(insufficient) if insufficient else None
if paired:
    lo, hi = boot_ci(paired)
    bar = max(KILL_BAR, statistics.median(floors))
    md += [f"success (< 2 A): {a.base} {succ[a.base]}, {a.test} {succ[a.test]}",
           f"PAIRED mean over complexes {statistics.mean(paired):+.3f} A, 95 % CI [{lo:+.3f}, {hi:+.3f}]",
           f"DEV median {statistics.median(devs):.3f} A vs bar max(0.60, FLOOR median {statistics.median(floors):.3f})"
           f" = {bar:.3f} A"]
    ok = statistics.median(devs) <= bar and succ[a.test] >= succ[a.base] - 1 and lo <= 0
    verdict = verdict or ("PASS" if ok else "FAIL")
md.append(f"VERDICT: {verdict}")
print("\n".join(md))
if a.out:
    a.out.write_text("\n".join(md) + "\n")

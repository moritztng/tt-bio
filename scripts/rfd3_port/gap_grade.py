#!/usr/bin/env python3
"""Grade RFD3 lever arms against the exact arm on the page fixture (host only, no device).

    gap_grade.py DIR [--ref=exact] [--n-design=100]

DIR holds <arm>_s<seed>.cif from gap_screen.py. For every arm and seed with a ref CIF of the same seed:

  drift      CA RMSD (Kabsch) to the ref arm, same seed, over the designed residues (the last N_DESIGN CA)
             and over the whole complex; the target is motif-fixed, so its CA stay put
  floor      the same RMSD between two different seeds of the ref arm: how far a design moves when only
             the seed changes. A lever's drift is read beside it, never alone
  quality    per CIF, over the designed residues, with the release gate's own checker
             (perf/wh-correctness/check_structure.py): CA-CA steps in band, breaks, heavy-atom clashes,
             distinct amino acids, top amino-acid fraction, UNK count
"""
import importlib.util
import itertools
import json
import pathlib
import re
import sys

import gemmi
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
D = pathlib.Path(args[0])
REF = opts.get("ref", "exact")
N_DES = int(opts.get("n-design", 100))

spec = importlib.util.spec_from_file_location("cs", ROOT / "perf" / "wh-correctness" / "check_structure.py")
geom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geom)


def kabsch(p, q):
    p, q = p - p.mean(0), q - q.mean(0)
    u, _s, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((p @ r - q) ** 2).sum(1).mean()))


def load(cif):
    st = gemmi.read_structure(str(cif))
    st.setup_entities()
    ca, names = [], []
    for ch in st[0]:
        for res in ch:
            a = res.find_atom("CA", "*")
            if a is not None:
                ca.append([a.pos.x, a.pos.y, a.pos.z])
                names.append(res.name)
    return st, np.asarray(ca), names


def quality(st, ca, names):
    des = ca[-N_DES:]
    d = np.linalg.norm(des[1:] - des[:-1], axis=1)
    lo, hi = geom.CA_CA_BAND
    seq = [geom._one_letter(n) for n in names[-N_DES:]]
    n_clash, heavy, _worst = geom.clashes(st)
    return {"in_band": round(float(((d >= lo) & (d <= hi)).mean()), 4),
            "breaks": int((d > geom.CA_CA_BREAK).sum()), "clashes": n_clash,
            "distinct_aa": len({c for c in seq if c != "X"}), "unk": seq.count("X"),
            "top_aa": round(max(seq.count(c) for c in set(seq)) / len(seq), 3)}


cifs = {}
for p in sorted(D.glob("*_s*.cif")):
    m = re.match(r"(.+)_s(\d+)\.cif$", p.name)
    cifs[(m.group(1), int(m.group(2)))] = load(p)
arms = sorted({a for a, _ in cifs}, key=lambda a: (a != REF, a))
out = {"ref": REF, "n_design": N_DES, "arms": {}, "floor": []}
ref_seeds = sorted(s for a, s in cifs if a == REF)
for s1, s2 in itertools.combinations(ref_seeds, 2):
    a, b = cifs[(REF, s1)][1], cifs[(REF, s2)][1]
    out["floor"].append({"seeds": [s1, s2], "design": kabsch(a[-N_DES:], b[-N_DES:]), "all": kabsch(a, b)})
for arm in arms:
    rows = []
    for (a, s), (st, ca, names) in sorted(cifs.items()):
        if a != arm:
            continue
        row = {"seed": s, **quality(st, ca, names)}
        if (REF, s) in cifs and arm != REF:
            rc = cifs[(REF, s)][1]
            row["drift_design"] = round(kabsch(ca[-N_DES:], rc[-N_DES:]), 3)
            row["drift_all"] = round(kabsch(ca, rc), 3)
        rows.append(row)
    out["arms"][arm] = rows
for arm, rows in out["arms"].items():
    for r in rows:
        print("%-12s s%-4d drift design %6s all %6s | in_band %.3f breaks %d clashes %d distinct %d top %.2f unk %d"
              % (arm, r["seed"], r.get("drift_design", "-"), r.get("drift_all", "-"), r["in_band"], r["breaks"],
                 r["clashes"], r["distinct_aa"], r["top_aa"], r["unk"]))
for f in out["floor"]:
    print("floor %s s%d vs s%d: design %.3f all %.3f" % (REF, *f["seeds"], f["design"], f["all"]))
(D / "grade.json").write_text(json.dumps(out, indent=1))

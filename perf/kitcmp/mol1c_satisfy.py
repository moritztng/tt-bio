"""Constraint satisfaction of predicted structures, upstream's geometry (docs/constraint_semantics.md): SITE residue
min heavy-atom distance to the partner chains <= 4.5 A, CONTACT pair <= 4.5 A, EXCLUSION residue > 6.0 A.
Usage: mol1c_satisfy.py <cases.json> <cif>... ; a CIF is matched to its case by the case name in its path.
Prints one JSON line per CIF: {"cif", "case", "site": [hits, n], "contact": [..], "exclusion": [..]}."""
import json
import sys

import numpy as np
import biotite.structure.io.pdbx as pdbx

CONTACT, EXCLUSION = 4.5, 6.0


def load(path):
    a = pdbx.get_structure(pdbx.CIFFile.read(path), model=1, use_author_fields=False)
    return a[a.element != "H"]


def mind(x, y):
    return float(np.min(np.linalg.norm(x[:, None] - y[None], axis=-1))) if len(x) and len(y) else float("inf")


def residue(a, c, r):
    return a.coord[(a.chain_id == c) & (a.res_id == r)]


def token_atoms(a, t):
    """Atoms of featurized token t: one per standard residue, one per atom of a ligand, in chain order."""
    keys = []
    for i in range(len(a)):
        k = (a.chain_id[i], a.res_id[i]) if a.res_name[i] in STD else (a.chain_id[i], a.res_id[i], i)
        if not keys or keys[-1][0] != k:
            keys.append((k, []))
        keys[-1][1].append(i)
    return a.coord[keys[t][1]]


STD = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL DA DC DG DT A C G U".split())


def score(a, cons):
    out = {}
    for kind, ok in (("site", lambda d: d <= CONTACT), ("exclusion", lambda d: d > EXCLUSION)):
        u = cons.get(kind)
        if u and u.get("residue_ids"):
            partner = a.coord[np.isin(a.chain_id, u["partner_chains"])]
            ds = [mind(residue(a, c, r), partner) for c, r in u["residue_ids"]]
            out[kind] = [sum(ok(d) for d in ds), len(ds), [round(d, 2) for d in ds]]
    c = cons.get("contact")
    if c:
        if "pairs" in c:
            ds = [mind(residue(a, *e1), residue(a, *e2)) for e1, e2 in c["pairs"]]
        else:
            ds = [mind(token_atoms(a, i), token_atoms(a, j)) for i, j in c["token_pairs"]]
        out["contact"] = [sum(d <= CONTACT for d in ds), len(ds), [round(d, 2) for d in ds]]
    return out


if __name__ == "__main__":
    cases = {c["name"]: c.get("constraints") or {} for c in json.load(open(sys.argv[1]))}
    for cif in sys.argv[2:]:
        name = max((n for n in cases if n in cif), key=len, default=None)
        if name is None:
            continue
        print(json.dumps({"cif": cif, "case": name, **score(load(cif), cases[name])}))

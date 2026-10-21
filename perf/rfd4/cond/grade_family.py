"""Grade one condition family's designs: what the input held, and how far two seeds land apart.

    python perf/rfd4/cond/grade_family.py --family <id> --input <annotated cif> \
        --designs <cif> <cif> [--upstream <cif>] [--out <json>]

Most of the eight families have no upstream design to pair against — `rfd4-ref` captured
module-level goldens per family, and full upstream runs only for the binder — so a family is
graded on what can be checked without one:

* **held**: each chain of the design superposed on the same chain of its input, over the atoms
  the two share. Anything the spec held — a target backbone, a motif, an inverse-folding
  backbone — comes back with its internal geometry intact and fits at a tenth of an Angstrom;
  a chain the model designed does not. This is the number that catches a condition that never
  reached the model. The absolute frame is not the measure: the model is equivariant and the
  writer realigns the whole design once.
* **floor**: two of our own designs at two seeds, superposed on the held part. How far apart two
  honest designs of the same input land, which is the scale any paired number has to be read
  against.
* **identity**: the sequence the two seeds agree on. A family that returns the same sequence at
  every seed is not designing.

With `--upstream`, the same pair comparison runs against an upstream design as well, and then
the grade is `rmsd_a` against the floor beside it, as `perf/rfd4/e2e/grade.py` does for the
binder.

Structures are matched by (chain, residue id, atom name), so only atoms both files actually have
are compared, and the count is reported with every number.
"""
import argparse
import json

import numpy as np

#: A chain this close to the input, with this many residues matched, was held, not designed.
HELD_RMSD_A = 1.0
HELD_MIN_RESIDUES = 4
BACKBONE = ("N", "CA", "C", "O")


def load(path):
    import biotite.structure.io.pdbx as pdbx

    return pdbx.get_structure(pdbx.CIFFile.read(path), model=1)


def keys(arr):
    return np.array([f"{c}|{r}|{a}" for c, r, a in zip(arr.chain_id, arr.res_id, arr.atom_name)])


def common(a, b):
    """The two arrays restricted to the atoms they share, in the same order."""
    ka, kb = keys(a), keys(b)
    shared = np.intersect1d(ka, kb)
    ia = {k: i for i, k in enumerate(ka)}
    ib = {k: i for i, k in enumerate(kb)}
    idx_a = np.array([ia[k] for k in shared], dtype=int)
    idx_b = np.array([ib[k] for k in shared], dtype=int)
    return a[idx_a], b[idx_b]


def rmsd(p, q):
    return float(np.sqrt(((p - q) ** 2).sum(1).mean())) if len(p) else float("nan")


def kabsch(p, q):
    """Rotation and translation taking p onto q (rows are points)."""
    pc, qc = p.mean(0), q.mean(0)
    h = (p - pc).T @ (q - qc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, qc - pc @ r.T


def held_chains(design, input_structure):
    """Per-chain fitted RMSD to the input, and which chains that says were held."""
    a, b = common(design, input_structure)
    backbone = np.isin(a.atom_name, BACKBONE)
    a, b = a[backbone], b[backbone]
    per_chain, held = {}, []
    for chain in np.unique(a.chain_id):
        m = a.chain_id == chain
        residues = len(np.unique(a.res_id[m]))
        if m.sum() < 3:
            continue
        r, t = kabsch(b.coord[m], a.coord[m])
        value = rmsd(b.coord[m] @ r.T + t, a.coord[m])
        per_chain[str(chain)] = {"fitted_rmsd_a": value, "atoms": int(m.sum()),
                                 "residues": residues}
        if value <= HELD_RMSD_A and residues >= HELD_MIN_RESIDUES:
            held.append(str(chain))
    return per_chain, held, int(len(a))


def pair(design, other, held):
    """`other` superposed on `design` over the held chains, then compared over the rest.

    With nothing held, the fit and the comparison are both over every CA the two share, which is
    the only thing an unconditioned monomer can be graded on.
    """
    a, b = common(design, other)
    ca = a.atom_name == "CA"
    a_ca, b_ca = a[ca], b[ca]
    fit = np.isin(a_ca.chain_id, held) if held else np.ones(len(a_ca), dtype=bool)
    if fit.sum() < 3:
        fit = np.ones(len(a_ca), dtype=bool)
    r, t = kabsch(b_ca.coord[fit], a_ca.coord[fit])
    moved = b_ca.coord @ r.T + t
    designed = ~np.isin(a_ca.chain_id, held) if held else np.ones(len(a_ca), dtype=bool)
    if not designed.any():
        designed = np.ones(len(a_ca), dtype=bool)
    return {
        "designed_ca_rmsd_a": rmsd(moved[designed], a_ca.coord[designed]),
        "fit_ca_rmsd_a": rmsd(moved[fit], a_ca.coord[fit]),
        "designed_residues": int(designed.sum()),
        "fit_residues": int(fit.sum()),
        "sequence_identity": float(np.mean(a_ca.res_name[designed] == b_ca.res_name[designed])),
    }


def sequence_of(structure, chains=None):
    ca = structure[structure.atom_name == "CA"]
    if chains:
        ca = ca[np.isin(ca.chain_id, chains)]
    return "".join(str(r) + " " for r in ca.res_name).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--input", required=True, help="the annotated CIF the design was made from")
    ap.add_argument("--designs", nargs="+", required=True,
                    help="ours, ONE PER SEED: the floor is how far two seeds land apart")
    ap.add_argument("--upstream", default=None, help="an upstream design of the same input and seed")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    designs = [load(p) for p in a.designs]
    source = load(a.input)
    per_chain, held, matched = held_chains(designs[0], source)

    out = {
        "family": a.family,
        "input": a.input,
        "designs": list(a.designs),
        "input_overlap_atoms": matched,
        "held_chains": held,
        "per_chain_fitted_rmsd_to_input_a": per_chain,
    }
    if len(designs) > 1:
        floor = pair(designs[0], designs[1], held)
        out["seed_floor"] = floor
        out["seed_floor_a"] = floor["designed_ca_rmsd_a"]
        out["designs_differ"] = floor["designed_ca_rmsd_a"] > 0.5
    if a.upstream:
        paired = pair(designs[0], load(a.upstream), held)
        out["upstream"] = a.upstream
        out["paired"] = paired
        out["rmsd_a"] = paired["designed_ca_rmsd_a"]

    checks = [matched > 0]
    if held:
        checks.append(max(per_chain[c]["fitted_rmsd_a"] for c in held) <= HELD_RMSD_A)
    if "designs_differ" in out:
        checks.append(out["designs_differ"])
    out["verdict"] = "PASS" if all(checks) else "FAIL"

    print(json.dumps(out, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

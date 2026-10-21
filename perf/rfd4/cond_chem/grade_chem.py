"""Grade an RFD4 design of a non-protein condition family against upstream's design.

    python perf/rfd4/cond_chem/grade_chem.py --ours <cif> --upstream <cif> --floor <cif> \
        [--designed <chain>] [--bar 0.60] [--out <json>]

`perf/rfd4/e2e/grade.py` grades the protein families and cannot grade these four. It picks the
designed chain as the shortest chain that has a `CA` atom, which breaks both ways here:

* a DNA/RNA chain has no `CA` at all, so a DNA-binder design leaves no target to superpose on and
  the grader raises on an empty concatenate;
* SAM has an atom *named* `CA`, so the ligand looks like a 1-residue chain and gets graded as the
  designed one while the real 150-residue design is treated as the target. That is a wrong number
  rather than a crash, which is worse.

So chains are classified by chemistry, not by atom name: biotite's own CCD-backed
`filter_amino_acids` / `filter_nucleotides`. The designed chain is the longest protein chain, or
the longest nucleic chain when there is no protein (RNA design conditioned on secondary
structure). Everything else is the condition and provides the superposition frame, which keeps
the contract `grade.py` uses: fit on what was held, then compare what was designed with no second
fit.

Atoms compared per chain: `CA` for protein, `C1'` for nucleic, every heavy atom for a ligand or
glycan. Atoms are matched by (chain, residue, atom name), never by position in the file.
"""
import argparse
import json

import numpy as np

HEAVY_EXCLUDE = {"H", "D"}


def load(path):
    import biotite.structure.io.pdbx as pdbx

    return pdbx.get_structure(pdbx.CIFFile.read(path), model=1)


def kabsch(p, q):
    """Rotation and translation taking p onto q (rows are points)."""
    pc, qc = p.mean(0), q.mean(0)
    h = (p - pc).T @ (q - qc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, qc - pc @ r.T


def chain_kinds(arr):
    """chain id -> 'protein' | 'nucleic' | 'other', by CCD residue class, not atom name."""
    import biotite.structure as struc

    aa, nuc = struc.filter_amino_acids(arr), struc.filter_nucleotides(arr)
    kinds = {}
    for c in np.unique(arr.chain_id):
        m = arr.chain_id == c
        n_aa, n_nuc = int((aa & m).sum()), int((nuc & m).sum())
        kinds[str(c)] = "protein" if n_aa > n_nuc and n_aa else "nucleic" if n_nuc else "other"
    return kinds


def n_residues(arr, chain):
    s = arr[arr.chain_id == chain]
    return len(np.unique(s.res_id))


def pick_designed(arr, kinds):
    """Longest protein chain; the longest nucleic chain when the design has no protein."""
    for kind in ("protein", "nucleic"):
        cand = [c for c, k in kinds.items() if k == kind]
        if cand:
            return max(cand, key=lambda c: n_residues(arr, c)), kind
    raise ValueError(f"no protein or nucleic chain to grade among {kinds}")


def rep_mask(arr, chain, kind):
    m = arr.chain_id == chain
    if kind == "protein":
        return m & (arr.atom_name == "CA")
    if kind == "nucleic":
        return m & (arr.atom_name == "C1'")
    return m & ~np.isin(arr.element, list(HEAVY_EXCLUDE))


def keyed(arr, mask):
    """(chain, res_id, atom_name) -> coordinate, for the masked atoms."""
    s = arr[mask]
    return {(str(c), int(r), str(a)): xyz
            for c, r, a, xyz in zip(s.chain_id, s.res_id, s.atom_name, s.coord)}


def paired(a, b):
    """Coordinates of the atoms the two structures share, in one common key order."""
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError("the two structures share no atom of this selection")
    return np.array([a[k] for k in keys]), np.array([b[k] for k in keys]), keys


def selections(ref, designed_override=None):
    kinds = chain_kinds(ref)
    if designed_override is not None:
        designed = designed_override
        if designed not in kinds:
            raise ValueError(f"chain {designed!r} not in {sorted(kinds)}")
        kind = kinds[designed]
    else:
        designed, kind = pick_designed(ref, kinds)
    anchors = [c for c in kinds if c != designed]
    return designed, kind, anchors, kinds


def compare(a, b, designed, kind, anchors, kinds):
    """Fit on the condition, then measure the design in that frame with no second fit."""
    if anchors:
        am = np.logical_or.reduce([rep_mask(a, c, kinds[c]) for c in anchors])
        bm = np.logical_or.reduce([rep_mask(b, c, kinds[c]) for c in anchors])
        pa, pb, anchor_keys = paired(keyed(a, am), keyed(b, bm))
        r, t = kabsch(pa, pb)
        anchor_rmsd = float(np.sqrt((((pa @ r.T + t) - pb) ** 2).sum(1).mean()))
        frame = "condition"
    else:
        # RNA design has nothing held: there is no condition frame, so the design is best-fit
        # onto its counterpart. That is a shape comparison, not a placement one.
        da0, db0, _ = paired(keyed(a, rep_mask(a, designed, kind)),
                             keyed(b, rep_mask(b, designed, kind)))
        r, t = kabsch(da0, db0)
        anchor_rmsd, anchor_keys, frame = None, [], "self (no condition held)"

    da, db, keys = paired(keyed(a, rep_mask(a, designed, kind)),
                          keyed(b, rep_mask(b, designed, kind)))
    da = da @ r.T + t
    sa = a[rep_mask(a, designed, kind)].res_name
    sb = b[rep_mask(b, designed, kind)].res_name
    return {
        "designed_chain": designed,
        "designed_kind": kind,
        "frame": frame,
        "anchor_chains": sorted(anchors),
        "anchor_atoms": len(anchor_keys),
        "anchor_rmsd_a": anchor_rmsd,
        "designed_atoms": len(keys),
        "designed_residues": int(len(np.unique([k[1] for k in keys]))),
        "rmsd_a": float(np.sqrt(((da - db) ** 2).sum(1).mean())),
        "sequence_identity": float(np.mean(sa == sb)) if len(sa) == len(sb) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--upstream", required=True, help="upstream's design at the same seed")
    ap.add_argument("--floor", required=True, help="upstream's design at another seed")
    ap.add_argument("--designed", default=None, help="override the designed chain id")
    ap.add_argument("--bar", type=float, default=0.60)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ours, up, floor = load(a.ours), load(a.upstream), load(a.floor)
    designed, kind, anchors, kinds = selections(up, a.designed)
    sel = (designed, kind, anchors, kinds)

    main_grade = compare(ours, up, *sel)
    rec = dict(main_grade)
    rec["chains"] = kinds
    rec["seed_floor_a"] = compare(up, floor, *sel)["rmsd_a"]
    rec["cross_a"] = compare(ours, floor, *sel)["rmsd_a"]
    rec["bar_a"] = a.bar
    rec["ours"], rec["upstream"], rec["floor"] = a.ours, a.upstream, a.floor

    r, f, x = rec["rmsd_a"], rec["seed_floor_a"], rec["cross_a"]
    # The bar is the verdict. Whether the paired trajectory held is a separate fact and is
    # reported as one: a design that re-rolled lands at the floor, and then the pair says
    # nothing on its own and only a distribution over many designs does.
    rec["verdict"] = "PASS" if r <= a.bar else "ABOVE BAR"
    rec["floor_ratio"] = r / f if f else None
    rec["pair_held"] = bool(f and x and r < 0.5 * min(f, x))

    print(json.dumps(rec, indent=2))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=2)


if __name__ == "__main__":
    main()

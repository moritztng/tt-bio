"""Grade a tt-bio RFD4 design against upstream's design of the same input and seed.

    python perf/rfd4/e2e/grade.py --ours <cif> --upstream <cif> --floor <upstream cif, another seed> \
        [--bar 0.60] [--out <json>]

All three structures are superposed on the target's CA atoms (the target is held by coordinate
conditioning, so that frame is the input's), then the designed chain is compared without a further
fit. Reported, all in Angstrom:

* `rmsd_a`: ours vs upstream, same seed, same noise (both sides seeded at the sampler's entry);
* `seed_floor_a`: upstream vs upstream at another seed, how far apart two honest designs land;
* `cross_a`: ours vs the other-seed upstream design. If `rmsd_a` is near `cross_a` and the floor, the
  trajectory re-rolled and the pair is two independent draws, so only a distributional grade over
  many designs can say anything; the script says so instead of passing or failing on it.

The designed chain is the one whose residues all carry the design marker; `--chain` overrides.
"""
import argparse
import json

import numpy as np


MARK_FIELD = "condition_sequence_1_residue"


def load(path):
    import biotite.structure.io.pdbx as pdbx

    f = pdbx.CIFFile.read(path)
    fields = [MARK_FIELD] if MARK_FIELD in f.block["atom_site"] else []
    return pdbx.get_structure(f, model=1, extra_fields=fields)


def kabsch(p, q):
    """Rotation and translation taking p onto q (rows are points)."""
    pc, qc = p.mean(0), q.mean(0)
    h = (p - pc).T @ (q - qc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, qc - pc @ r.T


def ca_by_chain(arr):
    ca = arr[arr.atom_name == "CA"]
    return {c: ca[ca.chain_id == c] for c in np.unique(ca.chain_id)}


def designed_and_target(ref, chain=None):
    chains = ca_by_chain(ref)
    if chain is not None:
        designed = chain
    else:
        # RFD4 writes the design marker into the sequence condition of every designed residue. Chain
        # length does not tell: the binder is 13 residues at 128 but 141 and 397 at 256 and 512.
        marked = [c for c, ca in chains.items()
                  if MARK_FIELD in ca.get_annotation_categories()
                  and (ca.get_annotation(MARK_FIELD) == "<M>").all()]
        if len(marked) != 1:
            raise ValueError(f"expected one chain carrying the design marker, got {marked}; pass --chain")
        designed = marked[0]
    targets = [c for c in chains if c != designed]
    return designed, targets


def compare(a, b, designed, targets):
    ca_a, ca_b = ca_by_chain(a), ca_by_chain(b)
    ta = np.concatenate([ca_a[c].coord for c in targets])
    tb = np.concatenate([ca_b[c].coord for c in targets])
    r, t = kabsch(ta, tb)
    target_rmsd = float(np.sqrt((((ta @ r.T + t) - tb) ** 2).sum(1).mean()))
    da, db = ca_a[designed].coord @ r.T + t, ca_b[designed].coord
    if da.shape != db.shape:
        raise ValueError(f"designed chain {designed}: {da.shape} vs {db.shape}")
    seq_a = "".join(ca_a[designed].res_name.astype(str))
    seq_b = "".join(ca_b[designed].res_name.astype(str))
    ident = float(np.mean(ca_a[designed].res_name == ca_b[designed].res_name))
    return {"designed_ca_rmsd_a": float(np.sqrt(((da - db) ** 2).sum(1).mean())),
            "target_ca_rmsd_a": target_rmsd, "designed_residues": int(len(db)),
            "sequence_identity": ident, "seq_a": seq_a, "seq_b": seq_b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--floor", required=True, help="upstream's design at another seed")
    ap.add_argument("--chain", default=None)
    ap.add_argument("--bar", type=float, default=0.60)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ours, up, other = load(a.ours), load(a.upstream), load(a.floor)
    designed, targets = designed_and_target(up, a.chain)
    pair = compare(ours, up, designed, targets)
    floor = compare(other, up, designed, targets)
    cross = compare(ours, other, designed, targets)
    rmsd, fl, cr = pair["designed_ca_rmsd_a"], floor["designed_ca_rmsd_a"], cross["designed_ca_rmsd_a"]
    if rmsd <= a.bar:
        verdict = "PASS"
    elif rmsd >= 0.5 * min(fl, cr):
        verdict = "REROLLED"  # the pair is two independent draws; grade by distribution instead
    else:
        verdict = "FAIL"
    out = {"designed_chain": designed, "target_chains": targets, "bar_a": a.bar,
           "rmsd_a": rmsd, "seed_floor_a": fl, "cross_a": cr,
           "pair": pair, "floor": floor, "cross": cross, "verdict": verdict,
           "ours": a.ours, "upstream": a.upstream, "floor_cif": a.floor}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k not in ("pair", "floor", "cross")}, indent=2))
    print("pair:", json.dumps(pair))


if __name__ == "__main__":
    main()

"""Score a predicted protein dimer against its crystal structure with DockQ.

  python score.py <crystal.cif[.gz]> <chainA>,<chainB> <pred.cif> [<pred.cif> ...]

The prediction's k-th chain is the crystal's k-th named chain; for identical chains both
assignments are tried and the better DockQ kept. Residues pair by position in the entity
sequence (label_seq_id), so unresolved crystal residues drop out. With one chain named, only
the CA RMSD is reported (the monomer control).

DockQ (Basu & Wallner 2016): fnat is the fraction of crystal residue contacts across the
interface (any heavy atom pair within 5 A) the prediction keeps; iRMS the backbone RMSD over
interface residues (within 10 A of the partner in the crystal) after fitting on them; LRMS the
backbone RMSD of the smaller chain after fitting on the larger. DockQ = mean of fnat,
1/(1+(iRMS/1.5)^2) and 1/(1+(LRMS/8.5)^2). >=0.80 high, >=0.49 medium, >=0.23 acceptable.
Given two or more predictions, the pairwise CA RMSD between them is the seed floor.
"""
import gzip
import io
import itertools
import sys

import biotite.structure.io.pdbx as pdbx
import numpy as np

BB = ("N", "CA", "C", "O")


def _load(path):
    text = gzip.open(path, "rt").read() if path.endswith(".gz") else open(path).read()
    a = pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(text)), model=1,
                           use_author_fields=False)
    return a[(a.element != "H") & ~a.hetero]


def _residues(a, cid):
    c = a[a.chain_id == cid]
    return {int(r): c[c.res_id == r] for r in np.unique(c.res_id)}


def _fit(x, y):
    """RMSD of y onto x after the optimal rigid fit, and the transform (R, cy, cx)."""
    cx, cy = x.mean(0), y.mean(0)
    u, _s, vt = np.linalg.svd((y - cy).T @ (x - cx))
    r = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    return float(np.sqrt((((y - cy) @ r + cx - x) ** 2).sum(1).mean())), (r, cy, cx)


def _rms(x, y, t):
    r, cy, cx = t
    return float(np.sqrt((((y - cy) @ r + cx - x) ** 2).sum(1).mean()))


def _common(xr, pr, names=None):
    """Matched atom coordinates (crystal, pred) of one residue, by atom name."""
    pn = set(pr.atom_name)
    keep = [n for n in xr.atom_name if n in pn and (names is None or n in names)]
    pick = lambda res: np.array([res.coord[res.atom_name == n][0] for n in keep]).reshape(-1, 3)
    return pick(xr), pick(pr)


def _chains(x, p, xc, pc):
    """Residues present in both, per chain: [(res_id, crystal_res, pred_res)]."""
    out = []
    for a, b in zip(xc, pc):
        xr, pr = _residues(x, a), _residues(p, b)
        out.append([(r, xr[r], pr[r]) for r in sorted(xr) if r in pr])
    return out


def _mind(a, b):
    return float(np.min(np.linalg.norm(a[:, None] - b[None], axis=-1)))


def dockq(x, p, xc, pc):
    rec, lig = _chains(x, p, xc, pc)
    if len(lig) > len(rec):
        rec, lig = lig, rec
    contacts, iface = set(), set()
    for i, (_r, xa, _pa) in enumerate(rec):
        for j, (_r2, xb, _pb) in enumerate(lig):
            d = _mind(xa.coord, xb.coord)
            if d < 5.0:
                contacts.add((i, j))
            if d < 10.0:
                iface |= {("r", i), ("l", j)}
    kept = 0
    for i, j in contacts:
        a, b = _common(rec[i][1], rec[i][2]), _common(lig[j][1], lig[j][2])
        kept += _mind(a[1], b[1]) < 5.0
    fnat = kept / len(contacts)
    bb = lambda res: [_common(xa, pa, BB) for _r, xa, pa in res]
    stack = lambda pairs, k: np.concatenate([q[k] for q in pairs])
    ib = bb([rec[i] for s, i in sorted(iface) if s == "r"]) + \
        bb([lig[j] for s, j in sorted(iface) if s == "l"])
    irms, _t = _fit(stack(ib, 0), stack(ib, 1))
    rb, lb = bb(rec), bb(lig)
    _r, t = _fit(stack(rb, 0), stack(rb, 1))
    lrms = _rms(stack(lb, 0), stack(lb, 1), t)
    dq = (fnat + 1 / (1 + (irms / 1.5) ** 2) + 1 / (1 + (lrms / 8.5) ** 2)) / 3
    ca = [_common(xa, pa, ("CA",)) for ch in (rec, lig) for _r, xa, pa in ch]
    rmsd, _t = _fit(stack(ca, 0), stack(ca, 1))
    return {"dockq": dq, "fnat": fnat, "irms": irms, "lrms": lrms, "rmsd": rmsd,
            "contacts": len(contacts), "ca": stack(ca, 1)}


def score(crystal, pred, xc):
    x, p = _load(crystal), _load(pred)
    pc = list(dict.fromkeys(p.chain_id.tolist()))
    assert len(pc) == len(xc), (pc, xc)
    if len(xc) == 1:
        (res,) = _chains(x, p, xc, pc)
        ca = [_common(xa, pa, ("CA",)) for _r, xa, pa in res]
        rmsd, _t = _fit(np.concatenate([q[0] for q in ca]), np.concatenate([q[1] for q in ca]))
        return {"rmsd": rmsd, "ca": np.concatenate([q[1] for q in ca])}
    seqs = [tuple(p.res_name[p.chain_id == c]) for c in pc]
    orders = [pc, pc[::-1]] if seqs[0] == seqs[1] else [pc]
    return max((dockq(x, p, xc, o) for o in orders), key=lambda r: r["dockq"])


def main():
    crystal, xc, preds = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
    rows = [score(crystal, pr, xc) for pr in preds]
    for pr, r in zip(preds, rows):
        if "dockq" in r:
            print(f"{pr}: DockQ {r['dockq']:.3f} fnat {r['fnat']:.2f} of {r['contacts']} "
                  f"iRMS {r['irms']:.2f} LRMS {r['lrms']:.2f} CA-RMSD {r['rmsd']:.2f} A")
        else:
            print(f"{pr}: CA-RMSD {r['rmsd']:.2f} A")
    for (i, a), (j, b) in itertools.combinations(enumerate(rows), 2):
        if len(a["ca"]) == len(b["ca"]):
            print(f"pairwise {preds[i]} vs {preds[j]}: {_fit(a['ca'], b['ca'])[0]:.2f} A")


if __name__ == "__main__":
    main()

"""Score a predicted protein-nucleic-acid complex against its crystal structure.

  python score.py <crystal.cif[.gz]> <pred.cif> [<pred.cif> ...]

Chains pair up in input order: the prediction's k-th chain is the crystal's k-th polymer chain
of the modelled entities (the input YAMLs list them in crystal order). Copies of one entity are
interchangeable, so every permutation among identical chains is tried and the lowest RMSD kept
(1LMB's repressor dimer can bind the operator with its two copies swapped). Residues pair by
position in the entity sequence (label_seq_id), so unresolved crystal residues drop out.

Reports, per prediction: whole-complex RMSD over CA (protein) and C1' (nucleotides) after one
optimal superposition, the RMSD of each molecule type under that same superposition, and the
interface contact recall: of the protein-nucleotide residue pairs within 5 A (any heavy atom)
in the crystal, the fraction also within 5 A in the prediction. Given two or more predictions,
also prints the pairwise RMSD between them, which is the seed floor.
"""
import gzip
import io
import itertools
import sys

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np

REP = {"CA", "C1'"}
CONTACT = 5.0


def _load(path):
    text = gzip.open(path, "rt").read() if path.endswith(".gz") else open(path).read()
    f = pdbx.CIFFile.read(io.StringIO(text))
    a = pdbx.get_structure(f, model=1, use_author_fields=False, extra_fields=[])
    return a[(a.element != "H") & ~a.hetero]


def _chains(a):
    out = []
    for cid in dict.fromkeys(a.chain_id.tolist()):
        c = a[a.chain_id == cid]
        out.append(c)
    return out


def _residues(chain):
    """label_seq_id -> atoms of that residue."""
    return {int(r): chain[chain.res_id == r] for r in np.unique(chain.res_id)}


def pair(crystal, pred, crystal_chains):
    """Matched (crystal_res, pred_res, is_protein, chain_k) quadruples."""
    xc = {c.chain_id[0]: c for c in _chains(crystal)}
    pc = _chains(pred)
    assert len(pc) == len(crystal_chains), (len(pc), crystal_chains)
    out = []
    for k, (cid, p) in enumerate(zip(crystal_chains, pc)):
        xr, pr = _residues(xc[cid]), _residues(p)
        for r, xa in xr.items():
            if r in pr:
                out.append((xa, pr[r], bool(np.isin("CA", xa.atom_name)), k))
    return out


def _rep(res):
    m = np.isin(res.atom_name, list(REP))
    return res.coord[m][0] if m.any() else None


def _superpose(x, y):
    """RMSD of y onto x after the optimal rigid fit, and the fitted y."""
    xc, yc = x - x.mean(0), y - y.mean(0)
    u, _s, vt = np.linalg.svd(yc.T @ xc)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    fit = yc @ r + x.mean(0)
    return float(np.sqrt(((fit - x) ** 2).sum(1).mean())), fit


def _contacts(res_list, coords_of):
    prot = [i for i, (_x, _p, isp, _k) in enumerate(res_list) if isp]
    nuc = [i for i, (_x, _p, isp, _k) in enumerate(res_list) if not isp]
    out = set()
    for i in prot:
        for j in nuc:
            a, b = coords_of(i), coords_of(j)
            if np.min(np.linalg.norm(a[:, None] - b[None], axis=-1)) < CONTACT:
                out.add((i, j))
    return out


def _seq(chain):
    return tuple(chain.res_name[chain.res_id == r][0] for r in np.unique(chain.res_id))


def score(crystal_path, pred_path, crystal_chains):
    """The best score over permutations of identical prediction chains."""
    crystal, pred = _load(crystal_path), _load(pred_path)
    seqs = [_seq(c) for c in _chains(pred)]
    groups = [[k for k, s in enumerate(seqs) if s == u] for u in dict.fromkeys(seqs)]
    best = None
    for perm in itertools.product(*(itertools.permutations(g) for g in groups)):
        order = list(range(len(seqs)))
        for g, pg in zip(groups, perm):
            for k, kk in zip(g, pg):
                order[k] = kk
        r = _score(crystal, pred, [crystal_chains[order[k]] for k in range(len(seqs))])
        if best is None or r["rmsd"] < best["rmsd"]:
            best = r
    return best


def _score(crystal, pred, crystal_chains):
    matched = pair(crystal, pred, crystal_chains)
    keep = [(x, p, isp, k) for x, p, isp, k in matched
            if _rep(x) is not None and _rep(p) is not None]
    X = np.array([_rep(x) for x, *_ in keep])
    Y = np.array([_rep(p) for _x, p, *_ in keep])
    rmsd, fit = _superpose(X, Y)
    isp = np.array([k[2] for k in keep])
    sub = lambda m: float(np.sqrt(((fit[m] - X[m]) ** 2).sum(1).mean()))
    # Common heavy atoms per residue, by name, for the contact test.
    def common(i, which):
        x, p = keep[i][0], keep[i][1]
        names = [n for n in x.atom_name if n in set(p.atom_name)]
        src = x if which == 0 else p
        return np.array([src.coord[src.atom_name == n][0] for n in names])
    native = _contacts(keep, lambda i: common(i, 0))
    predicted = _contacts(keep, lambda i: common(i, 1))
    recall = len(native & predicted) / len(native) if native else float("nan")
    return {"rmsd": rmsd, "rmsd_protein": sub(isp), "rmsd_nucleic": sub(~isp),
            "n_protein": int(isp.sum()), "n_nucleic": int((~isp).sum()),
            "contacts_native": len(native), "contact_recall": recall,
            "rep": Y, "keep": keep}


CRYSTAL_CHAINS = {"1LMB": ["A", "B", "C", "D"], "1URN": ["D", "A"]}


def main():
    crystal, preds = sys.argv[1], sys.argv[2:]
    key = next(k for k in CRYSTAL_CHAINS if k in crystal.upper())
    rows = [score(crystal, p, CRYSTAL_CHAINS[key]) for p in preds]
    for p, r in zip(preds, rows):
        print(f"{p}: CA/C1' RMSD {r['rmsd']:.2f} A (protein {r['rmsd_protein']:.2f} A over "
              f"{r['n_protein']}, nucleic {r['rmsd_nucleic']:.2f} A over {r['n_nucleic']}) | "
              f"interface contacts {r['contact_recall']:.2f} of {r['contacts_native']}")
    for (i, a), (j, b) in itertools.combinations(enumerate(rows), 2):
        if len(a["rep"]) == len(b["rep"]):
            print(f"pairwise {preds[i]} vs {preds[j]}: {_superpose(a['rep'], b['rep'])[0]:.2f} A")


if __name__ == "__main__":
    main()

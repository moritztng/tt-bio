#!/usr/bin/env python3
"""Score every Protenix prediction of the accuracy set against its deposited structure.

    venv/bin/python score.py <data dir> <run root> > preds.tsv

<run root> holds <arm>/pred/<PDB>/seed_<s>/predictions/<PDB>_sample_<k>.cif (+ _summary_confidence_sample_<k>.json),
with <arm> named <mode>_<seeds>. Chain 1 of every file is the target, chain 2 the binder; residues are matched
by their position in the deposited entity sequence (label_seq_id), so unmodelled reference residues just drop
out. Per prediction: DockQ v2 (fnat, iRMSD, LRMSD), TM-score of the complex, target and binder with the fixed
residue correspondence, and the model's own pLDDT / ipTM / pTM / ranking score.
"""
import json, re, sys, tempfile
from pathlib import Path

import gemmi
import numpy as np
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces


def chains(path):
    """[(label_seq -> residue)] for the first two polymer chains, protein only, hydrogens dropped."""
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    out = []
    for ch in st[0]:
        pol = ch.get_polymer()
        if len(pol):
            out.append({r.label_seq: r for r in pol if r.label_seq is not None})
    return out[:2]


def write_pdb(chs, keep, path):
    """The two chains as A/B, numbered by label_seq, only residues in `keep` (a pair of key sets)."""
    st = gemmi.Structure()
    m = gemmi.Model("1")
    for name, ch, ks in zip("AB", chs, keep):
        c = gemmi.Chain(name)
        for k in sorted(ks):
            r = gemmi.Residue()
            src = ch[k]
            r.name, r.seqid, r.entity_type = src.name, gemmi.SeqId(k, " "), gemmi.EntityType.Polymer
            for a in src:
                if a.altloc in ("\0", "A"):
                    b = gemmi.Atom(); b.name, b.element, b.pos, b.occ, b.b_iso = a.name, a.element, a.pos, 1.0, a.b_iso
                    r.add_atom(b)
            c.add_residue(r)
        m.add_chain(c)
    st.add_model(m)
    st.setup_entities()
    st.write_pdb(str(path))


def ca(ch, ks):
    return np.array([ch[k]["CA"][0].pos.tolist() for k in ks])


def kabsch(P, Q):
    """Rotation+translation that best maps P onto Q (least squares)."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return R, qc - pc @ R


def tm_score(P, Q, Lnorm):
    """TM-score with fixed correspondence (the TMscore program's search: seeds of L/1..L/8 fragments, iterated)."""
    L = len(P)
    if L < 3:
        return float("nan")
    d0 = max(1.24 * (max(Lnorm, 19) - 15) ** (1 / 3) - 1.8, 0.5)
    best = 0.0
    for frac in (1, 2, 4, 8):
        n = max(L // frac, 4)
        for s in range(0, L - n + 1, max(n // 2, 1)):
            idx = np.arange(s, s + n)
            for _ in range(20):
                R, t = kabsch(P[idx], Q[idx])
                d = np.linalg.norm(P @ R + t - Q, axis=1)
                best = max(best, float(np.sum(1 / (1 + (d / d0) ** 2)) / Lnorm))
                new = np.where(d < d0 + 1.5)[0]
                if len(new) < 3 or np.array_equal(new, idx):
                    break
                idx = new
    return best


def score(pred, ref, tmp, n):
    p, r = chains(pred), chains(ref)
    keys = [sorted(set(pc) & set(rc)) for pc, rc in zip(p, r)]
    # DockQ's load_PDB is cached by path: every pair gets its own file names.
    pp, rp = Path(tmp) / f"m{n}.pdb", Path(tmp) / f"n{n}.pdb"
    write_pdb(p, keys, pp)
    write_pdb(r, keys, rp)
    res, _ = run_on_all_native_interfaces(load_PDB(str(pp)), load_PDB(str(rp)), chain_map={"A": "A", "B": "B"})
    iface = next(iter(res.values()))
    P = [ca(p[i], keys[i]) for i in range(2)]
    Q = [ca(r[i], keys[i]) for i in range(2)]
    full_r = [len(rc) for rc in r]
    return dict(dockq=iface["DockQ"], fnat=iface["fnat"], irmsd=iface["iRMSD"], lrmsd=iface["LRMSD"],
                tm_complex=tm_score(np.vstack(P), np.vstack(Q), sum(full_r)),
                tm_target=tm_score(P[0], Q[0], full_r[0]), tm_binder=tm_score(P[1], Q[1], full_r[1]))


def main():
    data, root = Path(sys.argv[1]), Path(sys.argv[2])
    cols = ["pdb", "mode", "seed", "sample", "ranking_score", "iptm", "ptm", "plddt",
            "dockq", "fnat", "irmsd", "lrmsd", "tm_complex", "tm_target", "tm_binder"]
    print("\t".join(cols), flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        for n, cif in enumerate(sorted(root.glob("*/pred/*/seed_*/predictions/*_sample_*.cif"))):
            m = re.search(r"/(\w+?)_[\d-]+/pred/(\w+)/seed_(\d+)/predictions/\2_sample_(\d+)\.cif$", str(cif))
            mode, pdb, seed, k = m.groups()
            conf = json.loads((cif.parent / f"{pdb}_summary_confidence_sample_{k}.json").read_text())
            row = dict(pdb=pdb, mode=mode, seed=seed, sample=k, **{c: conf.get(c) for c in ("ranking_score", "iptm", "ptm", "plddt")},
                       **score(cif, data / "ref" / f"{pdb}.cif", tmp, n))
            print("\t".join(f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols), flush=True)


if __name__ == "__main__":
    main()

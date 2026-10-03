#!/usr/bin/env python3
"""How close each recorded fold came to the experimental structure.

    python3 demo/sc26/gallery/accuracy.py        # writes accuracy.json, read by build.py

Compares the final frame in store/ (the scored structure) with the PDB entry named in picks.json.
Residues are matched per chain by sequence alignment, so tags, missing loops and the GFP
chromophore (three residues fused into one in the crystal) drop out instead of shifting the
register. Reports C-alpha RMSD after one rigid superposition of the whole complex, the same per
chain, and the share of C-alpha atoms within 2 A. For hemes: heavy-atom RMSD of each heme after
the protein superposition, matched by atom name. Reference files are fetched from RCSB once into
runs/ref/; the demo itself never needs them.
"""
import json
import lzma
import urllib.request
from pathlib import Path

import gemmi
import numpy as np
from Bio.Align import PairwiseAligner

HERE = Path(__file__).resolve().parent


def final_xyz(meta):
    raw = lzma.decompress((HERE / "store" / meta["bin"]["file"]).read_bytes())
    return np.frombuffer(raw[-meta["n_atoms"] * 12:], "<f4").reshape(-1, 3).astype(np.float64)


def ref_structure(pdb):
    path = HERE / "runs" / "ref" / f"{pdb}.cif"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb}.cif", path)
    st = gemmi.read_structure(str(path))
    st.remove_alternative_conformations()
    return st


def kabsch(p, q):
    pc, qc = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((p - pc).T @ (q - qc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, qc - r @ pc


def rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


def main():
    al = PairwiseAligner(mode="global", match_score=2, mismatch_score=-1, open_gap_score=-6,
                         extend_gap_score=-0.5)
    out = {}
    for pick in json.load(open(HERE / "picks.json")):
        mp = HERE / "store" / f"{pick['id']}.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        xyz = final_xyz(meta)
        at = meta["atoms"]
        ref = ref_structure(pick["pdb"])[0]
        P, Q, per_chain = [], [], {}
        for c in pick["chains"]:
            ca = {}
            for i, (ch, nm, r) in enumerate(zip(at["chain"], at["name"], at["residue"])):
                if ch == c["id"] and nm == "CA":
                    ca[r] = xyz[i]
            pred = [ca[r] for r in sorted(ca)]
            assert len(pred) == len(c["sequence"]), (pick["id"], c["id"], len(pred))
            rch = ref[c["ref_chain"]]
            rres = [r for r in rch if (ti := gemmi.find_tabulated_residue(r.name)) and ti.is_amino_acid()
                    and r.find_atom("CA", "*") is not None]
            rseq = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper() or "X" for r in rres)
            aln = al.align(c["sequence"], rseq)[0]
            p, q = [], []
            for (a0, a1), (b0, b1) in zip(*aln.aligned):
                for k in range(a1 - a0):
                    if c["sequence"][a0 + k] == rseq[b0 + k]:
                        p.append(pred[a0 + k])
                        q.append(np.array(rres[b0 + k].find_atom("CA", "*").pos.tolist()))
            p, q = np.array(p), np.array(q)
            r, t = kabsch(p, q)
            per_chain[c["id"]] = dict(role=c["role"], ref_chain=c["ref_chain"], matched=len(p),
                                      of=len(c["sequence"]), ca_rmsd=round(rmsd(p @ r.T + t, q), 2))
            P.append(p)
            Q.append(q)
        P, Q = np.concatenate(P), np.concatenate(Q)
        r, t = kabsch(P, Q)
        d = np.sqrt(((P @ r.T + t - Q) ** 2).sum(1))
        res = dict(pdb=pick["pdb"], matched_ca=len(P), ca_rmsd=round(float(np.sqrt((d ** 2).mean())), 2),
                   ca_within_2A=round(float((d < 2).mean()), 3), per_chain=per_chain)
        hemes = []
        for lig in pick["ligands"]:
            pred = {n: xyz[i] for i, (ch, n) in enumerate(zip(at["chain"], at["name"])) if ch == lig["id"]}
            pc = np.mean(list(pred.values()), 0) @ r.T + t
            best = None
            for ch in ref:
                for rr in ch:
                    if rr.name == lig["ccd"]:
                        refa = {a.name: np.array(a.pos.tolist()) for a in rr}
                        common = [n for n in pred if n in refa]
                        moved = np.array([pred[n] for n in common]) @ r.T + t
                        target = np.array([refa[n] for n in common])
                        dist = float(np.linalg.norm(target.mean(0) - pc))
                        if best is None or dist < best[0]:
                            best = (dist, rmsd(moved, target), len(common))
            hemes.append(dict(ligand=lig["id"], ccd=lig["ccd"], atoms=best[2], rmsd=round(best[1], 2)))
        if hemes:
            res["ligands"] = hemes
        out[pick["id"]] = res
        print(pick["id"], json.dumps(res))
    (HERE / "accuracy.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

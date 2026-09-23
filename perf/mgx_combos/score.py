#!/usr/bin/env python3
"""Score a combination fold against its crystal: what came back, and is it the complex.

score.py <crystal.cif[.gz]> <pred.cif> [...]  prints one line per prediction:
polymer chains and ligand atoms returned, whole-complex CA-RMSD after superposition (chains of one
sequence matched in whichever order fits best), each ligand's centroid distance to the nearest
crystal copy of the same CCD code, and the closest ligand-protein heavy-atom contact (< 1 A is a
clash). --json writes the records instead.
"""
import itertools
import json
import sys

import gemmi
import numpy as np


def _polymers(st):
    out = {}
    for ch in st[0]:
        pol = ch.get_polymer()
        if len(pol) < 2:
            continue
        seq = gemmi.one_letter_code([r.name for r in pol])
        ca = [(r.seqid.num, r.find_atom("CA", "*")) for r in pol]
        out[ch.name] = (seq, [(i, a.pos) for i, (n, a) in enumerate(ca) if a])
    return out


def _ligands(st):
    out = []
    for ch in st[0]:
        for r in ch:
            if r.het_flag == "H" and r.name not in ("HOH", "GOL", "EDO", "SO4", "PEG"):
                xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r if a.element.name != "H"])
                if len(xyz):
                    out.append((ch.name, r.name, xyz))
    return out


def _pairs(ps, pc):
    """CA pairs of one predicted chain against one crystal chain, by sequence alignment."""
    (sp, cap), (sc, cac) = ps, pc
    al = gemmi.align_string_sequences(list(sp), list(sc), [])
    ip = ic = 0
    mp, mc = dict(cap), dict(cac)
    out = []
    for op, n in al.cigar_tuples() if hasattr(al, "cigar_tuples") else _cigar(al.cigar_str()):
        for _ in range(n):
            if op == "M":
                if ip in mp and ic in mc:
                    out.append((mp[ip], mc[ic]))
                ip += 1
                ic += 1
            elif op == "I":
                ip += 1
            else:
                ic += 1
    return out, al.calculate_identity()


def _cigar(s):
    num = ""
    for c in s:
        if c.isdigit():
            num += c
        else:
            yield c, int(num)
            num = ""


def _kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, pc, qc


def score(crystal, pred):
    cst = gemmi.read_structure(crystal)
    cst.setup_entities()
    pst = gemmi.read_structure(pred)
    pst.setup_entities()
    cp, pp = _polymers(cst), _polymers(pst)
    # candidate crystal chains for each predicted chain: identity >= 0.9
    cand = {}
    for pn, ps in pp.items():
        cand[pn] = [cn for cn, cs in cp.items() if _pairs(ps, cs)[1] >= 90]
    best = None
    names = list(pp)
    for combo in itertools.product(*[cand[n] or [None] for n in names]):
        used = [c for c in combo if c]
        if len(used) != len(set(used)):
            continue
        P, Q = [], []
        for pn, cn in zip(names, combo):
            if cn:
                for a, b in _pairs(pp[pn], cp[cn])[0]:
                    P.append([a.x, a.y, a.z])
                    Q.append([b.x, b.y, b.z])
        if len(P) < 3:
            continue
        P, Q = np.array(P), np.array(Q)
        R, pc, qc = _kabsch(P, Q)
        rmsd = float(np.sqrt((((P - pc) @ R.T + qc - Q) ** 2).sum(1).mean()))
        if best is None or rmsd < best[0]:
            best = (rmsd, len(P), R, pc, qc)
    rec = {"pred": pred, "chains": {n: len(s) for n, (s, _) in pp.items()}}
    pl, cl = _ligands(pst), _ligands(cst)
    rec["ligands"] = {f"{c}:{n}": len(x) for c, n, x in pl}
    if best:
        rmsd, n, R, pc, qc = best
        rec["ca_rmsd"], rec["n_ca"] = round(rmsd, 3), n
        prot = np.array([[a.pos.x, a.pos.y, a.pos.z] for ch in pst[0] for r in ch.get_polymer()
                         for a in r if a.element.name != "H"])
        lig = {}
        for c, nm, x in pl:
            xs = (x - pc) @ R.T + qc
            same = [y for _, m, y in cl if m == nm]
            d = min((float(np.linalg.norm(xs.mean(0) - y.mean(0))) for y in same), default=None)
            contact = float(np.min(np.linalg.norm(x[:, None] - prot[None], axis=2))) if len(prot) else None
            lig[f"{c}:{nm}"] = {"centroid_to_crystal": None if d is None else round(d, 2),
                                "min_contact": None if contact is None else round(contact, 2)}
        rec["ligand_fit"] = lig
    return rec


def main():
    args = [a for a in sys.argv[1:] if a != "--json"]
    recs = [score(args[0], p) for p in args[1:]]
    if "--json" in sys.argv:
        print(json.dumps(recs, indent=1))
        return
    for r in recs:
        lig = " ".join(f"{k}({r['ligands'][k]} at, {v['centroid_to_crystal']} A, contact {v['min_contact']})"
                       for k, v in r.get("ligand_fit", {}).items())
        print(f"{r['pred']}: chains {r['chains']} CA-RMSD {r.get('ca_rmsd')} A over {r.get('n_ca')} | {lig}")


if __name__ == "__main__":
    main()

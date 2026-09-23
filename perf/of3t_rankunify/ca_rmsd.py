"""Ca-RMSD of a predicted structure against an experimental one, chains matched by sequence.

Kabsch superposition over the Ca atoms, `tt_bio.align.rmsd`, the same maths the OpenFold3 fold
gate uses. Chains are paired by one-letter sequence: the abag/ubq yaml targets are built from
the resolved residues of their reference so the pairing is exact, and where a reference chain
is longer the common prefix of its residue ids is used. A target whose chains cannot be paired
1:1 raises rather than reporting a number against the wrong chain.
"""
from __future__ import annotations

import numpy as np
import torch
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb

from tt_bio import align


def _load(path: str):
    if str(path).endswith(".pdb"):
        f = pdb.PDBFile.read(str(path))
        a = pdb.get_structure(f, model=1)
    else:
        a = pdbx.get_structure(pdbx.CIFFile.read(str(path)), model=1)
    a = a[struc.filter_amino_acids(a) & (a.atom_name == "CA") & (a.element != "H")]
    return a


def _chains(a):
    out = {}
    for c in struc.chain_iter(a):
        ids, names = struc.get_residues(c)
        out[str(c.chain_id[0])] = (ids, names, c.coord.astype(np.float64))
    return out


def ca_rmsd(pred_path: str, ref_path: str) -> float:
    p, r = _chains(_load(pred_path)), _chains(_load(ref_path))
    if len(p) != len(r):
        raise ValueError(f"chain count {len(p)} != {len(r)} ({pred_path} vs {ref_path})")
    used, pairs = set(), []
    for pk, (pid, pnm, pxyz) in p.items():
        best = None
        for rk, (rid, rnm, rxyz) in r.items():
            if rk in used:
                continue
            n = min(len(pnm), len(rnm))
            ident = float(np.mean(pnm[:n] == rnm[:n])) if n else 0.0
            if best is None or ident > best[0]:
                best = (ident, rk, n)
        ident, rk, n = best
        if ident < 0.8:
            raise ValueError(f"no sequence match for chain {pk} (best {ident:.2f})")
        used.add(rk)
        pairs.append((pxyz[:n], r[rk][2][:n]))
    P = np.concatenate([x for x, _ in pairs])
    R = np.concatenate([y for _, y in pairs])
    return float(align.rmsd(torch.from_numpy(P), torch.from_numpy(R)))

#!/usr/bin/env python3
"""Global CA RMSD between two predicted CIFs, Kabsch, paired by residue index.

Unpaired cross-stack RMSD carries the full seed floor, so this is only readable next
to a same-engine different-seed control -- which is why this row runs one.
"""
import sys
import gemmi, numpy as np

def ca(path):
    st = gemmi.read_structure(path); st.setup_entities()
    out = []
    for ch in st[0]:
        for r in ch:
            a = r.find_atom('CA', '*')
            if a: out.append([a.pos.x, a.pos.y, a.pos.z])
    return np.array(out)

A, B = ca(sys.argv[1]), ca(sys.argv[2])
assert A.shape == B.shape, (A.shape, B.shape)
A = A - A.mean(0); B = B - B.mean(0)
U, S, Vt = np.linalg.svd(A.T @ B)
d = np.sign(np.linalg.det(U @ Vt))
R = U @ np.diag([1, 1, d]) @ Vt
print(f'{sys.argv[1].split("/")[-1]} vs {sys.argv[2].split("/")[-1]}: '
      f'n={len(A)} CA-RMSD={np.sqrt((((A @ R) - B) ** 2).sum(1).mean()):.3f} A')

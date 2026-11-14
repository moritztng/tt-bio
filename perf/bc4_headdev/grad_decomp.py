#!/usr/bin/env python3
"""Where a card gradient's error lives: norms, the per-row mean component, the worst rows and columns.
usage: grad_decomp.py <sm_card --dump npz>"""
import sys

import numpy as np

d = np.load(sys.argv[1])
for nm in ("d_act0", "d_pair"):
    r, g = d["ref_" + nm], d[nm]
    e = g - r
    print(f"{nm}: |ref| {np.linalg.norm(r):.4e} |card| {np.linalg.norm(g):.4e} |err| {np.linalg.norm(e):.4e} "
          f"rel {np.linalg.norm(e) / np.linalg.norm(r):.3e} cos {np.sum(r * g) / np.linalg.norm(r) / np.linalg.norm(g):.4f}")
    flat_e = e.reshape(-1, e.shape[-1])
    m = flat_e.mean(-1, keepdims=True)
    print(f"  per-row-mean share of err {np.linalg.norm(np.broadcast_to(m, flat_e.shape)) / np.linalg.norm(flat_e):.3f}")
    rows = np.linalg.norm(flat_e, axis=-1)
    print(f"  worst rows {np.argsort(rows)[-5:][::-1].tolist()} share of err in top 5 rows "
          f"{np.sqrt((np.sort(rows)[-5:] ** 2).sum()) / np.linalg.norm(flat_e):.3f}")
    cols = np.linalg.norm(flat_e, axis=0)
    print(f"  worst cols {np.argsort(cols)[-5:][::-1].tolist()} share in top 5 cols "
          f"{np.sqrt((np.sort(cols)[-5:] ** 2).sum()) / np.linalg.norm(flat_e):.3f}")
a = d["act0"]
print(f"act0 |row| median {np.median(np.linalg.norm(a, axis=-1)):.3e} max {np.abs(a).max():.3e}")

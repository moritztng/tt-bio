#!/usr/bin/env python3
"""Phase 4 label: pairwise DockQ + TM-score matrices + PSS over a fold's sample ensemble.

For a given (target, generator) fold, computes the N x N (N = diffusion_samples)
pairwise matrix of inter-sample DockQ (global DockQ, reusing the DockQ lib with chain-map
by sequence, allowed_mismatches=0) and TM-score (via the tmtools package, which wraps the
US-align reference implementation). PSS (Predicted Structure Score, mandatory per FROZEN
DESIGN) = mean pairwise interface DockQ across the ensemble (the upper-triangle mean),
a single scalar summarising ensemble self-consistency.

Usage:
    PYTHONPATH=<worktree> python3 scripts/abag_xm_pairwise_matrix.py <results_dir> <target> [--n_samples 50] [--out json]

<results_dir>/structures/<target>.cif is the rank-0 winner; <target>_model_<k>.cif are
ranks 1..N-1. The script reads all N sample CIFs and computes the upper-triangle matrix.
"""
import argparse, json, sys
from pathlib import Path
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces, group_chains, get_all_chain_maps


def _dockq(cif_a, cif_b):
    """Global DockQ of cif_a vs cif_b (cif_b treated as native). Symmetric enough for a matrix."""
    ms = load_PDB(cif_a)
    ns = load_PDB(cif_b)
    mc, nc = [c.id for c in ms], [c.id for c in ns]
    clusters, rev = group_chains(ms, ns, mc, nc, allowed_mismatches=0)
    try:
        cmap = next(get_all_chain_maps(clusters, {}, rev, mc, nc))
    except StopIteration:
        return None
    res, total = run_on_all_native_interfaces(ms, ns, chain_map=cmap)
    return (total / len(res)) if res else 0.0


import numpy as np
THREE_TO_ONE = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M","SEC":"U","PYL":"O","UNK":"X"}

def _ca_coords_and_seq(cif):
    """Return (Nx3 float64 CA coords, sequence string) over the whole complex in chain order."""
    import gemmi
    st = gemmi.read_structure(str(cif))
    coords, seq = [], []
    for model in st:
        for ch in model:
            for res in ch.first_conformer():
                for at in res:
                    if at.name == "CA":
                        coords.append([at.pos.x, at.pos.y, at.pos.z])
                        seq.append(THREE_TO_ONE.get(res.name.strip().upper(), "X"))
                        break
        break
    return np.asarray(coords, dtype=np.float64), "".join(seq)

def _tm(cif_a, cif_b):
    import tmtools
    xa, sa = _ca_coords_and_seq(cif_a)
    xb, sb = _ca_coords_and_seq(cif_b)
    r = tmtools.tm_align(xa, xb, sa, sb)
    return float(r.tm_norm_chain1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("target")
    ap.add_argument("--n_samples", type=int, default=50)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sdir = Path(a.results_dir) / "structures"
    cifs = [sdir / f"{a.target}.cif"]  # rank 0 (winner)
    for k in range(1, a.n_samples):
        cifs.append(sdir / f"{a.target}_model_{k}.cif")
    cifs = [c for c in cifs if c.exists()]
    n = len(cifs)
    if n < 2:
        print(json.dumps({"target": a.target, "n_samples": n, "error": "need >=2 sample CIFs"}))
        sys.exit(1)

    rows = []
    dockq_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            dq = _dockq(str(cifs[i]), str(cifs[j]))
            tm = _tm(cifs[i], cifs[j])
            rows.append({"i": i, "j": j, "dockq": round(dq, 6) if dq is not None else None,
                          "tm": round(tm, 6)})
            if dq is not None:
                dockq_vals.append(dq)
            print(f"[{i},{j}] dockq={rows[-1]['dockq']} tm={rows[-1]['tm']}", file=sys.stderr)

    pss = sum(dockq_vals) / len(dockq_vals) if dockq_vals else None
    out = {"target": a.target, "n_samples": n,
           "PSS": round(pss, 6) if pss is not None else None,
           "matrix": rows}
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

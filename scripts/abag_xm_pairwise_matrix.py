#!/usr/bin/env python3
"""Phase 4 label: pairwise DockQ + TM-score matrices + PSS over a fold's sample ensemble.

D6/D7: PSS = mean pairwise INTERFACE DockQ on the ARK-declared Ab-Ag interface
(fold_auth_chain_id_1/2), NOT the auto-mapper GlobalDockQ average over all native
interfaces. The average includes the internal Fab H-L interface (always near-1.0),
which inflates PSS and destroys its discriminative power. So we call run_on_chains
on ONLY the two declared chains (antibody + antigen), mirroring
abag_xm_dockq_interface.py.

Perf: each sample CIF is parsed ONCE (DockQ structure + CA coords/seq cached); the
declared interface's model chain ids are resolved once (rank-0 vs native by
sequence) and reused across all pairs. The O(N^2) loop then reuses cached objects.

Usage:
    PYTHONPATH=<wt> python3 scripts/abag_xm_pairwise_matrix.py <results_dir> <target> [--n_samples 50] [--out json] [--chain1 C1] [--chain2 C2]

<results_dir>/structures/<target>.cif is rank-0; <target>_model_<k>.cif are ranks 1..N-1.
chain1/chain2 default to the manifest fold_auth_chain_id_1/2 (resolved by pdb_id).
"""
import argparse, json, sys
from pathlib import Path
from DockQ.DockQ import (load_PDB, run_on_chains, calc_DockQ,
                          align_chains, format_alignment,
                          group_chains, get_all_chain_maps)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "docs" / "implementation-parity-data" / "abag-xm-targets.parquet"
GT = ROOT / "examples" / "ground_truth_structures"


def _declared_chains(target):
    try:
        import pandas as pd
        df = pd.read_parquet(MANIFEST)
        row = df[df["pdb_id"] == target]
        if len(row) == 0:
            return None, None
        r = row.iloc[0]
        return r["fold_auth_chain_id_1"], r["fold_auth_chain_id_2"]
    except Exception:
        return None, None


def _chain_obj(struct, cid):
    for c in struct:
        if c.id == cid:
            return c
    raise KeyError(f"chain {cid!r} not in {[c.id for c in struct]}")


def _resolve_model_chains(ms0, ns, c1, c2):
    """Map declared native chains (c1,c2) to model chain ids via sequence (group_chains).
    DockQ get_all_chain_maps returns {native_id: model_id}. Returns (m_ag, m_ab) or None.
    Any chain-resolution failure (e.g. group_chains raising on a mismatched/odd layout
    like 9iar) degrades gracefully to None -> pairwise runs without precomputed
    alignments (slower run_on_chains per pair) instead of crashing the whole matrix."""
    try:
        mc = [c.id for c in ms0]
        nc = [c.id for c in ns]
        clusters, rev = group_chains(ms0, ns, mc, nc, allowed_mismatches=0)
        cmap = next(get_all_chain_maps(clusters, {}, rev, mc, nc))   # {native: model}
        inv = {m: n for n, m in cmap.items()}
        out = {}
        for declared, role in ((c1, "c1"), (c2, "c2")):
            if declared in nc:
                out[role] = cmap[declared]
            elif declared in mc:
                out[role] = declared
            else:
                return (None, None)
        return out["c1"], out["c2"]
    except Exception:
        return (None, None)


def _dockq_pair(cached_i, cached_j, m1, m2, precomputed_alignments):
    """calc_DockQ on the two declared model chains with a PRE-COMPUTED alignment.

    All samples in one fold share IDENTICAL chain sequences (only coordinates
    differ), so align_chains (the per-pair sequence alignment run_on_chains does
    internally) is identity work -- the residue mapping is the same for every pair.
    We pre-compute it once (rank-0 vs rank-0) and call calc_DockQ directly, bypassing
    align_chains per pair. This is the difference between ~1s/pair and ~0.1s/pair
    (the dominant cost for O(N^2) matrices). For small_molecule (het) chains we
    fall back to run_on_chains (the calc_sym_corrected_lrmsd path needs the live
    alignment and is rare for Ab-Ag)."""
    mi = (_chain_obj(cached_i["struct"], m1), _chain_obj(cached_i["struct"], m2))
    mj = (_chain_obj(cached_j["struct"], m1), _chain_obj(cached_j["struct"], m2))
    sm = bool(getattr(mi[0], "is_het", False) or getattr(mi[1], "is_het", False) or
             getattr(mj[0], "is_het", False) or getattr(mj[1], "is_het", False))
    if sm or precomputed_alignments is None:
        info = run_on_chains(mi, mj, small_molecule=sm)
        return None if info is None else info.get("DockQ")
    info = calc_DockQ(mi, mj, alignments=precomputed_alignments)
    return None if info is None else info.get("DockQ")


def _precompute_alignments(cached0, m1, m2):
    """Identity alignment for the two declared chains (rank-0 vs rank-0).
    Returns tuple of (seqA, matches, seqB) per chain, or None on failure."""
    try:
        alns = []
        for cid in (m1, m2):
            c = _chain_obj(cached0["struct"], cid)
            aln = align_chains(c, c)              # identity (same sequence)
            alns.append(tuple(format_alignment(aln).values()))
        return tuple(alns)
    except Exception:
        return None


THREE_TO_ONE = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M","SEC":"U","PYL":"O","UNK":"X"}


def _ca_coords_and_seq(cif):
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
    import numpy as np
    return np.asarray(coords, dtype=np.float64), "".join(seq)


def _tm_pair(cached_i, cached_j):
    import tmtools
    r = tmtools.tm_align(cached_i["ca"], cached_j["ca"], cached_i["seq"], cached_j["seq"])
    return float(r.tm_norm_chain1)




_WORKER_CACHE = None

def _init_worker(cif_paths, m1, m2, precomputed_alignments):
    """Each worker parses all CIFs once into a process-local cache."""
    global _WORKER_CACHE
    cache = []
    for c in cif_paths:
        st = load_PDB(str(c))
        ca, seq = _ca_coords_and_seq(c)
        cache.append({"struct": st, "ca": ca, "seq": seq})
    _WORKER_CACHE = (cache, m1, m2, precomputed_alignments)

def _pair_worker(ij):
    i, j = ij
    cache, m1, m2, aln = _WORKER_CACHE
    dq = None
    if m1 is not None and m2 is not None:
        dq = _dockq_pair(cache[i], cache[j], m1, m2, aln)
    tm = _tm_pair(cache[i], cache[j])
    return (i, j, (round(dq, 6) if dq is not None else None), round(tm, 6))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("target")
    ap.add_argument("--n_samples", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--chain1", default=None)
    ap.add_argument("--chain2", default=None)
    ap.add_argument("--n_workers", type=int, default=4,
                       help="parallel pair workers (0=serial); 4 is ~3.5x on large folds")
    a = ap.parse_args()

    sdir = Path(a.results_dir) / "structures"
    cifs = [sdir / f"{a.target}.cif"]
    for k in range(1, a.n_samples):
        cifs.append(sdir / f"{a.target}_model_{k}.cif")
    cifs = [c for c in cifs if c.exists()]
    n = len(cifs)
    if n < 2:
        print(json.dumps({"target": a.target, "n_samples": n,
                          "error": "need >=2 sample CIFs"}))
        sys.exit(1)

    c1, c2 = a.chain1, a.chain2
    if c1 is None or c2 is None:
        mc1, mc2 = _declared_chains(a.target)
        c1 = c1 or mc1
        c2 = c2 or mc2

    # Cache each sample's parsed structure + CA coords/seq (parse ONCE, reuse).
    cached = []
    for c in cifs:
        st = load_PDB(str(c))
        ca, seq = _ca_coords_and_seq(c)
        cached.append({"struct": st, "ca": ca, "seq": seq})

    # Resolve declared interface -> model chain ids once (rank-0 vs native).
    native = GT / f"{a.target}.cif"
    if c1 is not None and c2 is not None and native.exists():
        ns = load_PDB(str(native))
        m1, m2 = _resolve_model_chains(cached[0]["struct"], ns, c1, c2)
        if m1 is None:
            print(json.dumps({"target": a.target, "n_samples": n,
                              "error": f"could not resolve declared chains {c1},{c2}",
                              "chain1": c1, "chain2": c2}), file=sys.stderr)
            m1, m2 = None, None
    else:
        m1, m2 = None, None

    rows = []
    dockq_vals = []
    precomputed_alignments = None
    if m1 is not None and m2 is not None:
        precomputed_alignments = _precompute_alignments(cached[0], m1, m2)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if a.n_workers and a.n_workers > 1 and len(pairs) >= a.n_workers:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.n_workers, initializer=_init_worker,
                      initargs=(cifs, m1, m2, precomputed_alignments)) as pool:
            results = pool.map(_pair_worker, pairs, chunksize=max(1, len(pairs) // (a.n_workers * 4)))
        for i, j, dq, tm in results:
            rows.append({"i": i, "j": j, "dockq": dq, "tm": tm})
            if dq is not None:
                dockq_vals.append(dq)
            print(f"[{i},{j}] dockq={dq} tm={tm}", file=sys.stderr)
    else:
        for i, j in pairs:
            if m1 is not None and m2 is not None:
                dq = _dockq_pair(cached[i], cached[j], m1, m2, precomputed_alignments)
            else:
                dq = None
            tm = _tm_pair(cached[i], cached[j])
            rows.append({"i": i, "j": j,
                          "dockq": round(dq, 6) if dq is not None else None,
                          "tm": round(tm, 6)})
            if dq is not None:
                dockq_vals.append(dq)
            print(f"[{i},{j}] dockq={rows[-1]['dockq']} tm={rows[-1]['tm']}",
                  file=sys.stderr)

    pss = sum(dockq_vals) / len(dockq_vals) if dockq_vals else None
    out = {"target": a.target, "n_samples": n,
           "interface": (f"{c1}_{c2}" if c1 and c2 else None),
           "model_chain1": m1, "model_chain2": m2,
           "PSS": round(pss, 6) if pss is not None else None,
           "matrix": rows}
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

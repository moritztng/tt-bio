#!/usr/bin/env python3
"""Score SC26 GFP folds: mean pLDDT and C-alpha RMSD to a crystal, per chip and per CPU seed.

    python perf/sc26_gfp/score.py --pdb 1EMA --chips runs/gfp_chips_r1.jsonl --ref runs/gfp_cpu_fp32.json

Residues are matched by sequence alignment exactly as demo/sc26/gallery/accuracy.py does, so the
GFP chromophore (three residues fused into one in the crystal) drops out instead of shifting the
register. Chip folds are read from a demo client log (fold_start carries the atoms, fold_done the
final coordinates and per-residue pLDDT). Also prints the largest coordinate difference between
chips: the demo folds every chip at seed 0, so any nonzero value is output that depends on the card.
"""
import argparse
import base64
import json
import urllib.request
from pathlib import Path

import gemmi
import numpy as np
from Bio.Align import PairwiseAligner

HERE = Path(__file__).resolve().parent
AL = PairwiseAligner(mode="global", match_score=2, mismatch_score=-1, open_gap_score=-6,
                     extend_gap_score=-0.5)


def crystal_ca(pdb, chain="A"):
    path = HERE / "runs" / "ref" / f"{pdb}.cif"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb}.cif", path)
    st = gemmi.read_structure(str(path))
    st.remove_alternative_conformations()
    res = [r for r in st[0][chain] if (t := gemmi.find_tabulated_residue(r.name)) and t.is_amino_acid()
           and r.find_atom("CA", "*") is not None]
    seq = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper() or "X" for r in res)
    return seq, np.array([r.find_atom("CA", "*").pos.tolist() for r in res])


def kabsch(p, q):
    pc, qc = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((p - pc).T @ (q - qc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, qc - r @ pc


def ca_rmsd(seq, ca, rseq, rca):
    aln = AL.align(seq, rseq)[0]
    i, j = [], []
    for (a0, a1), (b0, b1) in zip(*aln.aligned):
        for k in range(a1 - a0):
            if seq[a0 + k] == rseq[b0 + k]:
                i.append(a0 + k)
                j.append(b0 + k)
    p, q = np.asarray(ca)[i], rca[j]
    r, t = kabsch(p, q)
    d = np.sqrt(((p @ r.T + t - q) ** 2).sum(1))
    return float(np.sqrt((d ** 2).mean())), len(i), float((d < 2).mean())


def pair_rmsd(a, b):
    a, b = np.asarray(a), np.asarray(b)
    r, t = kabsch(a, b)
    return float(np.sqrt(((a @ r.T + t - b) ** 2).sum(1).mean()))


def chip_folds(path):
    starts, out = {}, {}
    for line in open(path):
        m = json.loads(line)
        if m.get("source") != "live":
            continue
        if m["type"] == "fold_start":
            starts[m["id"]] = m
        elif m["type"] == "fold_done" and m["id"] in starts:
            s = starts[m["id"]]
            xyz = np.frombuffer(base64.b64decode(m["xyz"]), "<f4").reshape(-1, 3).astype(np.float64)
            ca = xyz[[i for i, n in enumerate(s["atoms"]["name"]) if n == "CA"]]
            out[m["id"]] = dict(chip=m["chip"], seq=s["sequence"], seed=s["seed"], ca=ca, xyz=xyz,
                                plddt=np.array(m["plddt"]), seconds=m["seconds"], aiclk=m["aiclk_mhz"],
                                ptm=m["ptm"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="1EMA")
    ap.add_argument("--chips", nargs="*", default=[])
    ap.add_argument("--ref", nargs="*", default=[])
    ap.add_argument("--seq", help="which chip folds to score when no --ref names the sequence")
    args = ap.parse_args()
    rseq, rca = crystal_ca(args.pdb)
    report, ref_ca = {"pdb": args.pdb}, {}
    for path in args.ref:
        r = json.load(open(path))
        for seed, run in sorted(r["runs"].items()):
            rmsd, n, w2 = ca_rmsd(r["sequence"], run["ca"], rseq, rca)
            ref_ca[f"{r['dtype']}:{seed}"] = np.array(run["ca"])
            print(f"REF {r['dtype']} seed {seed}: pLDDT {run['plddt_mean']:.3f} pTM {run['ptm']:.4f} "
                  f"CA RMSD {rmsd:.2f} A ({n} CA, {w2:.0%} within 2 A)")
    folds = {}
    for path in args.chips:
        folds.update(chip_folds(path))
    want = json.load(open(args.ref[0]))["sequence"] if args.ref else args.seq
    folds = {k: f for k, f in folds.items() if f["seq"] == want}  # the log also holds other folds
    for jid, f in sorted(folds.items(), key=lambda kv: kv[1]["chip"]):
        rmsd, n, w2 = ca_rmsd(f["seq"], f["ca"], rseq, rca)
        vs = " ".join(f"vs-{k} {pair_rmsd(f['ca'], v):.2f}" for k, v in ref_ca.items())
        print(f"CHIP {f['chip']} {jid} seed {f['seed']}: pLDDT {f['plddt'].mean():.3f} pTM {f['ptm']:.4f} "
              f"CA RMSD {rmsd:.2f} A ({n} CA, {w2:.0%} within 2 A) {f['seconds']} s "
              f"AICLK {f['aiclk']} {vs}")
    xs = list(folds.values())
    if len(xs) > 1:
        dx = max(float(np.abs(a["xyz"] - xs[0]["xyz"]).max()) for a in xs[1:])
        dp = max(float(np.abs(a["plddt"] - xs[0]["plddt"]).max()) for a in xs[1:])
        print(f"ACROSS CHIPS: max |dxyz| {dx:.3g} A, max |dpLDDT| {dp:.3g} over {len(xs)} folds")


if __name__ == "__main__":
    main()

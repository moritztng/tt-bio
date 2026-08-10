#!/usr/bin/env python3
"""Compare coords from two arms (base vs changed) — PCC + RMSD + bit-exact check."""
import sys, math
from pathlib import Path
import torch


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    num = (a * b).sum()
    den = a.norm() * b.norm()
    return float(num / den) if den > 0 else 0.0


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--changed", required=True)
    a = ap.parse_args()
    db = torch.load(a.base, map_location="cpu")
    dc = torch.load(a.changed, map_location="cpu")
    cb, cc = db["coords"], dc["coords"]
    print(f"base: arm={db['arm']} wall={db['wall_s']}s plddt={db['plddt']} shape={db['shape']}")
    print(f"changed: arm={dc['arm']} wall={dc['wall_s']}s plddt={dc['plddt']} shape={dc['shape']}")
    assert cb.shape == cc.shape, f"shape mismatch {cb.shape} vs {cc.shape}"
    bitexact = torch.equal(cb, cc)
    p = pcc(cb, cc)
    rmsd = float(((cb - cc) ** 2).mean().sqrt().item())
    maxabs = float((cb - cc).abs().max().item())
    print(f"PCC={p:.6f} RMSD={rmsd:.6f} maxabs={maxabs:.6f} bit_exact={bitexact}")
    print(f"plddt_delta={dc['plddt'] - db['plddt']:.6e}")
    out = Path(a.changed).with_suffix(".pcc.json")
    out.write_text(__import__("json").dumps({
        "base_arm": db["arm"], "changed_arm": dc["arm"],
        "base_wall_s": db["wall_s"], "changed_wall_s": dc["wall_s"],
        "base_plddt": db["plddt"], "changed_plddt": dc["plddt"],
        "pcc": p, "rmsd": rmsd, "maxabs": maxabs, "bit_exact": bitexact,
        "plddt_delta": dc["plddt"] - db["plddt"]}) + "\n", indent=2)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

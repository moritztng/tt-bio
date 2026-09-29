#!/usr/bin/env python3
"""Turn an RCSB download into the single clean target chain-set a design campaign takes.

Keeps the first model, the requested chains, protein ATOM records only, and the highest-
occupancy altloc of each atom. Waters, ligands and the crystallographic extras go: a
BindCraft 2 target is the receptor a binder is designed against, and a sulfate in the
lattice is neither. Residue numbering is left exactly as deposited, so a hotspot quoted
from the paper still lands on the residue the paper meant.

Prints one line per output: path, chains, residues per chain, and any gap in the deposited
numbering, because a chain break inside a target is a real input a researcher brings and
the next row needs to know which of these have one.
"""
import pathlib
import sys

AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
      "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE"}


def clean(src: pathlib.Path, dst: pathlib.Path, chains: str) -> dict:
    want = chains.split(",")
    best: dict = {}          # (chain, resseq, atom) -> (occupancy, line)
    order: list = []
    for line in src.read_text().splitlines():
        if line.startswith("ENDMDL"):
            break            # first model only
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resname = line[17:20].strip()
        if resname not in AA:
            continue         # protein only; MSE is selenomethionine, a real residue
        chain = line[21]
        if chain not in want:
            continue
        key = (chain, line[22:27], line[12:16].strip())
        occ = float(line[54:60] or 1.0)
        if key not in best:
            order.append(key)
        if key not in best or occ > best[key][0]:
            # ATOM, not HETATM, and one altloc: a downstream parser that sees two copies of
            # CA either picks one silently or refuses, and neither belongs in a size ladder.
            best[key] = (occ, "ATOM  " + line[6:16] + " " + line[17:])
    out, n = [], 0
    for key in order:
        n += 1
        out.append(f"{best[key][1][:6]}{n:5d}{best[key][1][11:]}")
    dst.write_text("\n".join(out) + "\nTER\nEND\n")

    per, gaps = {}, {}
    for chain in want:
        seq = [int(k[1][:4]) for k in order if k[0] == chain and k[2] == "CA"]
        per[chain] = len(seq)
        gaps[chain] = [(a, b) for a, b in zip(seq, seq[1:]) if b != a + 1]
    return {"path": str(dst), "chains": chains, "residues": sum(per.values()),
            "per_chain": per, "gaps": gaps}


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    for spec in sys.argv[1:]:
        pdb, chains, name = spec.split(":")
        info = clean(here / f"{pdb}.pdb", here / f"{name}.pdb", chains)
        print(f"{name}: {info['residues']} aa, chains {info['per_chain']}, "
              f"gaps {info['gaps']}")

#!/usr/bin/env python3
"""Characterise OpenDDE's broken chains: which residues break, and how.

`check_structure.py` reports "N backbone gaps > 5 A -- chain is broken", which reads
as a chain in N+1 pieces. It is not. Every gap in an OpenDDE artifact comes in an
adjacent pair around a *single* residue, and that residue's own backbone bonds are
collapsed to a fraction of their ideal lengths. So the defect is a small set of
residues whose atoms were never denoised, not a fragmented fold -- a different bug
with a different fix, and the distinction is invisible in the gap count alone.

    odde_ejected.py results/artifacts/size_128_opendde/target_1.cif [...]

Prints, per file: the ejected residue indices, their internal N-CA / CA-C / C-O
bond lengths against a neighbour's, and the fraction of the chain affected.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# Ideal backbone geometry. A residue the diffusion updated lands near these; one it
# skipped collapses well under them, which is the discriminator this script uses.
IDEAL = {"N-CA": 1.46, "CA-C": 1.52, "C-O": 1.23}
COLLAPSED = 1.2  # A: N-CA below this is not a real peptide bond
BREAK = 5.0      # A: the CA-CA step check_structure.py calls a gap


def parse(path: Path) -> tuple[dict[str, int], list[list[str]]]:
    """Column index -> row list for the `_atom_site` loop of an mmCIF."""
    cols: list[str] = []
    rows: list[list[str]] = []
    in_loop = False
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("_atom_site."):
            cols.append(s.split(".", 1)[1])
            in_loop = True
        elif in_loop and (s.startswith("ATOM") or s.startswith("HETATM")):
            rows.append(s.split())
        elif in_loop and s == "#":
            in_loop = False
    return {c: k for k, c in enumerate(cols)}, rows


def report(path: Path) -> None:
    i, rows = parse(path)
    name = lambda r: r[i["label_atom_id"]].strip('"')
    seq = lambda r: int(r[i["label_seq_id"]])
    xyz = lambda r: [float(r[i[c]]) for c in ("Cartn_x", "Cartn_y", "Cartn_z")]

    by_res: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        by_res.setdefault(seq(r), {})[name(r)] = xyz(r)
    ca = [(n, a["CA"]) for n, a in sorted(by_res.items()) if "CA" in a]

    steps = [math.dist(ca[k][1], ca[k + 1][1]) for k in range(len(ca) - 1)]
    broken = {k for k, d in enumerate(steps) if d > BREAK}
    # Residue k (0-based into `ca`) is ejected when the step before it and the step
    # after it are both breaks -- it is detached from the chain on both sides.
    ejected = [ca[k][0] for k in range(1, len(ca) - 1) if (k - 1) in broken and k in broken]

    in_band = sum(1 for d in steps if 3.6 <= d <= 4.1)
    print(f"{path}")
    print(f"  {len(ca)} residues | {100 * in_band / len(steps):.1f}% of steps in band "
          f"| {len(broken)} steps > {BREAK} A | {len(ejected)} ejected residues "
          f"({100 * len(ejected) / len(ca):.2f}%)")
    if not ejected:
        return
    print(f"  ejected: {ejected}")
    for n in ejected:
        mine = _bonds(by_res.get(n, {}))
        nbr = _bonds(by_res.get(n + 1, {}))
        flag = "COLLAPSED" if mine.get("N-CA", 9) < COLLAPSED else ""
        print(f"    {n:4d} {by_res[n].get('_', '')}{_fmt(mine)}  neighbour {n + 1}: {_fmt(nbr)} {flag}")
    print(f"  ideal: {_fmt(IDEAL)}")


def _bonds(atoms: dict[str, list[float]]) -> dict[str, float]:
    out = {}
    for a, b in (("N", "CA"), ("CA", "C"), ("C", "O")):
        if a in atoms and b in atoms:
            out[f"{a}-{b}"] = math.dist(atoms[a], atoms[b])
    return out


def _fmt(d) -> str:
    return " ".join(f"{k} {v:.2f}" for k, v in d.items())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    for p in sys.argv[1:]:
        report(Path(p))
        print()

#!/usr/bin/env python3
"""Strip hydrogens and deuteriums from an mmCIF's `_atom_site` loop.

PXDesign parses a user's CIF through protenix's `DistillationMMCIFParser`, whose pipeline is
`fix_arginine` then `add_missing_atoms_and_residues` -- the parent `MMCIFParser` runs
`Filter.remove_water` and `Filter.remove_hydrogens` first, and the subclass does not. So
hydrogens reach the step that matches observed atoms against the residue's CCD atom names,
where they all count as mismatches, and that step keeps a residue only when

    len(matched) > len(mismatched)                  # protenix/data/parser.py

A residue whose hydrogen count is at least its heavy-atom count therefore fails the test and
is marked fully unresolved, coordinates zeroed. The loss is by residue type, not by disorder:
ALA (5 heavy, 5 H), VAL (7, 9), LEU (8, 11), ILE (8, 11), LYS (9, 13), MET (8, 9), PRO (7, 7),
THR (7, 7) and ARG (11, 13) all fail; GLY, SER, CYS, ASN, ASP, GLN, GLU, HIS, PHE, TRP and TYR
all pass. On the shipped PD-L1 quick-start target (`examples/5o45.cif`, 1248 hydrogens) that
is 61 of 116 cropped residues, and PXDesign conditions the generator on them at the origin.

Run the target CIF through here before `pxdesign parse` and all 116 residues survive.

    python3 scripts/pxdesign_port/strip_cif_hydrogens.py in.cif out.cif
"""
from __future__ import annotations

import sys

ATOM_SITE = "_atom_site."


def strip(src: str, dst: str) -> tuple[int, int]:
    lines = open(src).read().splitlines()
    out: list[str] = []
    kept = dropped = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("loop_"):
            out.append(line)
            i += 1
            continue
        # a loop_ header runs until the first non-`_` line
        j = i + 1
        header = []
        while j < len(lines) and lines[j].startswith("_"):
            header.append(lines[j].strip())
            j += 1
        out.append(line)
        out.extend(lines[i + 1:j])
        if not any(h.startswith(ATOM_SITE) for h in header):
            i = j
            continue
        col = header.index(ATOM_SITE + "type_symbol")
        while j < len(lines) and lines[j].strip() and not lines[j].startswith("#"):
            if lines[j].split()[col] in ("H", "D"):
                dropped += 1
            else:
                out.append(lines[j])
                kept += 1
            j += 1
        i = j
    if not dropped and not kept:
        raise SystemExit(f"{src}: no _atom_site loop found")
    open(dst, "w").write("\n".join(out) + "\n")
    return kept, dropped


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__.strip().splitlines()[-1].strip())
    kept, dropped = strip(sys.argv[1], sys.argv[2])
    print(f"kept {kept} atoms, dropped {dropped} hydrogens -> {sys.argv[2]}")

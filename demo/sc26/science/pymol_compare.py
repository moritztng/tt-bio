#!/usr/bin/env python3
"""The final frame of a recording in PyMOL, cartoon coloured on the AlphaFold pLDDT scale, to set
next to the demo's own render of the same structure.

    python3 demo/sc26/science/pymol_compare.py <recording.jsonl> <out-prefix> [--ss OURS]

Writes <out-prefix>.pdb (pLDDT in the B-factor column) and <out-prefix>-pymol.png, and prints
PyMOL's secondary structure (its `dss`) per residue. With --ss (the demo's own H/E/C string, from
science/ss.mjs) it also prints how often the two agree. Needs a Python with PyMOL
(`pip install pymol-open-source-whl`); it is a check, not part of the booth.
"""
import argparse
import base64
import json

import numpy as np

THREE = dict(A="ALA", C="CYS", D="ASP", E="GLU", F="PHE", G="GLY", H="HIS", I="ILE", K="LYS", L="LEU",
             M="MET", N="ASN", P="PRO", Q="GLN", R="ARG", S="SER", T="THR", V="VAL", W="TRP", Y="TYR")
BANDS = [(90, "0x0053D6"), (70, "0x65CBF3"), (50, "0xFFDB13"), (0, "0xFF7D45")]


def write_pdb(path, start, done):
    xyz = np.frombuffer(base64.b64decode(done["xyz"]), "<f4").reshape(-1, 3)
    seq = start["sequence"].replace(":", "")
    plddt = done["plddt"]
    k = 100 if max(plddt) <= 1.01 else 1
    a = start["atoms"]
    with open(path, "w") as f:
        for i, (el, name, r) in enumerate(zip(a["element"], a["name"], a["residue"])):
            rn = THREE.get(seq[r], "UNK") if r < len(seq) else "LIG"
            rec = "ATOM  " if r < len(seq) else "HETATM"
            nm = name if len(name) == 4 else " " + name.ljust(3)
            f.write(f"{rec}{i + 1:5d} {nm} {rn} A{r + 1:4d}    {xyz[i, 0]:8.3f}{xyz[i, 1]:8.3f}{xyz[i, 2]:8.3f}"
                    f"  1.00{plddt[r] * k:6.2f}          {el:>2s}\n")
        f.write("END\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("out")
    ap.add_argument("--ss", default="")
    a = ap.parse_args()
    evs = [json.loads(l) for l in open(a.recording)]
    start = next(e for e in evs if e["type"] == "fold_start")
    done = next(e for e in evs if e["type"] == "fold_done")
    write_pdb(a.out + ".pdb", start, done)

    from pymol import cmd
    cmd.load(a.out + ".pdb", "m")
    cmd.dss("m")
    ss = {"": "C", "L": "C", "S": "E"}   # PyMOL calls a strand S
    got = []
    cmd.iterate("m and name CA", "got.append(ss)", space={"got": got})
    theirs = "".join(ss.get(s, s) for s in got)
    print("pymol dss", theirs)
    if a.ss:
        n = min(len(a.ss), len(theirs))
        agree = sum(x == y for x, y in zip(a.ss[:n], theirs[:n])) / n
        print(f"demo  ss  {a.ss}\nagreement {agree:.3f} over {n} residues")
    cmd.hide("everything")
    cmd.show("cartoon")
    cmd.show("spheres", "hetatm")
    for lo, hexcol in reversed(BANDS):   # low to high, each band over the one below
        cmd.color(hexcol, f"m and b > {lo}" if lo else "m")
    cmd.set("cartoon_fancy_helices", 0)
    cmd.bg_color("0x08090C")
    cmd.set("ray_opaque_background", 1)
    cmd.set("depth_cue", 1)
    cmd.orient("m")
    cmd.png(a.out + "-pymol.png", width=1600, height=1200, dpi=150, ray=1)


if __name__ == "__main__":
    main()

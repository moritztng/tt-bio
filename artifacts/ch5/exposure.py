"""Which peptide residues face out of the HLA groove, by burial count.

For each residue of the peptide chain, count heavy atoms of the MHC heavy chain and beta-2m
within 5 A of its side chain (CA onward; glycine uses CA). A low count means the side chain
points up out of the groove and is reachable by a binder; a high count means it is anchored
into the MHC and invisible from outside. This is what decides whether D6 is actually a usable
discriminator or is buried.
"""
import math, sys, pathlib

CUT = 5.0
PEP = "C"
MHC = ("A", "B")
BACKBONE = {"N", "C", "O", "OXT"}

src = pathlib.Path(sys.argv[1])
lines = src.read_text().splitlines()
start = next(i for i, l in enumerate(lines) if l.startswith("_atom_site."))
i, cols = start, []
while lines[i].startswith("_atom_site."):
    cols.append(lines[i].strip().split(".", 1)[1]); i += 1
ci = {c: n for n, c in enumerate(cols)}

pep, mhc = [], []
while i < len(lines) and not lines[i].startswith("#"):
    f = lines[i].split(); i += 1
    if len(f) < len(cols) or f[ci["group_PDB"]] != "ATOM":
        continue
    if f[ci["type_symbol"]] == "H":
        continue
    ch = f[ci["auth_asym_id"]]
    xyz = (float(f[ci["Cartn_x"]]), float(f[ci["Cartn_y"]]), float(f[ci["Cartn_z"]]))
    if ch == PEP:
        pep.append((int(f[ci["auth_seq_id"]]), f[ci["label_comp_id"]], f[ci["label_atom_id"]], xyz))
    elif ch in MHC:
        mhc.append(xyz)

order, names, counts = [], {}, {}
for num, comp, atom, (x, y, z) in pep:
    if num not in counts:
        order.append(num); names[num] = comp; counts[num] = 0
    if atom in BACKBONE:
        continue
    for (mx, my, mz) in mhc:
        if (x - mx) ** 2 + (y - my) ** 2 + (z - mz) ** 2 <= CUT * CUT:
            counts[num] += 1

# Glycine has no side chain, so its count is near zero for a trivial reason and says nothing
# about exposure. Rank only the residues that have a side chain to place.
real = [(n, num) for n, num in enumerate(order, 1) if names[num] != "GLY"]
lo = min(counts[num] for _, num in real)
hi = max(counts[num] for _, num in real)

print("peptide residue burial (MHC heavy atoms within {} A of the side chain)".format(CUT))
for n, num in enumerate(order, 1):
    if names[num] == "GLY":
        print("  P{:<3} {:<4} {:>4}  (no side chain, count not meaningful)".format(
            n, names[num], counts[num]))
        continue
    bar = "#" * int(round(20 * (counts[num] - lo) / max(1, hi - lo)))
    print("  P{:<3} {:<4} {:>4}  {}".format(n, names[num], counts[num], bar))

ranked = sorted(real, key=lambda t: counts[t[1]])
print("side-chain residues, least to most buried: "
      + ", ".join("P{}={}".format(n, counts[num]) for n, num in ranked))

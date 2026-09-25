"""Non-adjacent CA-CA pairs closer than 3 A -- the instrument ws:mgx-ceilings reported boltz2 as
having "no CA-CA gap over 4.2 A" on. check_structure counts HEAVY-ATOM contacts under 2 A, a
different question, and comparing one number against the other is how a fixture property reads as
a regression. Columns are resolved from the loop_ header rather than by position.
"""
import sys
import numpy as np

for path in sys.argv[1:]:
    cols, xyz, inloop = {}, [], False
    for line in open(path):
        s = line.strip()
        if s.startswith("_atom_site."):
            cols[s.split(".", 1)[1]] = len(cols); inloop = True; continue
        if inloop and (s.startswith("ATOM") or s.startswith("HETATM")):
            f = s.split()
            if f[cols["label_atom_id"]] == "CA" and f[cols["group_PDB"]] == "ATOM":
                xyz.append([float(f[cols["Cartn_" + c]]) for c in "xyz"])
        elif inloop and xyz and s.startswith("#"):
            break
    a = np.array(xyz)
    d = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
    i, j = np.triu_indices(len(a), k=2)          # k=2 skips the bonded neighbour
    adj = np.abs(np.diff(a, axis=0))
    step = np.linalg.norm(np.diff(a, axis=0), axis=-1)
    print(f"{path.split(chr(47))[-1]}: {len(a)} CA, "
          f"{int((d[i, j] < 3.0).sum())} non-adjacent pairs < 3.0 A, min {d[i, j].min():.3f} A, "
          f"max consecutive CA-CA step {step.max():.3f} A, "
          f"{int((step > 4.2).sum())} steps > 4.2 A")

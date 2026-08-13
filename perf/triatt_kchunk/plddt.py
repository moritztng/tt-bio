import sys, glob, os
import numpy as np
for d in sys.argv[1:]:
    for p in sorted(glob.glob(f"{d}/intermediate_designs/*.cif")):
        hdr, vals, inloop = [], [], False
        for line in open(p):
            if line.startswith("_atom_site."):
                hdr.append(line.strip().split(".")[1]); inloop = True; continue
            if inloop and line.startswith(("ATOM", "HETATM")):
                f = dict(zip(hdr, line.split()))
                if f.get("label_atom_id") == "CA":
                    vals.append(float(f["B_iso_or_equiv"]))
            elif inloop and line.startswith("#"):
                inloop = False
        v = np.array(vals)
        print(f"{os.path.basename(d)}/{os.path.basename(p)}: n_CA={len(v)} "
              f"mean_plDDT={v.mean():.4f} median={np.median(v):.4f} min={v.min():.4f}")

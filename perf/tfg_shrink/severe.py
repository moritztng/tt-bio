"""Severe antibody-antigen overlaps (d < 0.75 (ra+rb), rdkit vdW radii, heavy atoms) per CIF, as upstream's rigid search defines them.
usage: severe.py LABEL GLOB [LABEL GLOB ...]; movable = the target's contact JSON movable_chains."""
import sys, glob, json, os, re, collections, numpy as np
R = dict(C=1.7, N=1.6, O=1.55, S=1.8, SE=1.9, P=1.8)
def movable(p):
    t = re.search(r"/([0-9a-z]{4})_(contact|pocket|unconstrained)", p).group(1)
    d = json.load(open(f"/home/moritz/tfg-ref/panel/{t}/{t}_contact.json")); d = d[0] if isinstance(d, list) else d
    return set(d["constraint"]["movable_chains"])
def atoms(p):
    mv = movable(p)
    hdr = []; mov = []; fix = []
    for l in open(p):
        if l.startswith("_atom_site."): hdr.append(l.strip().split(".")[1])
        elif l.startswith(("ATOM", "HETATM")):
            r = l.split(); ix = {h: i for i, h in enumerate(hdr)}
            el = r[ix["type_symbol"]].upper()
            if el == "H": continue
            row = [float(r[ix[k]]) for k in ("Cartn_x", "Cartn_y", "Cartn_z")] + [R.get(el, 1.7)]
            (mov if r[ix["label_asym_id"]] in mv else fix).append(row)
    return np.array(mov), np.array(fix)
if __name__ == "__main__":
    for label, pat in zip(sys.argv[1::2], sys.argv[2::2]):
        fs = sorted(glob.glob(pat, recursive=True)); counts = []
        for f in fs:
            m, x = atoms(f)
            d = np.linalg.norm(m[:, None, :3] - x[None, :, :3], axis=-1)
            counts.append(int((d < 0.75 * (m[:, None, 3] + x[None, :, 3])).sum()))
        c = np.array(counts)
        print(f"{label}: n={len(c)} with>=1 severe {np.mean(c>0):.2f}  mean pairs {c.mean():.2f}  p90 {np.percentile(c,90):.0f}  max {c.max()}", flush=True)

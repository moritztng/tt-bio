"""Severe overlaps between the two antibody chains (heavy-light interface), unguided: a control for interface clash that no guidance touches."""
import sys, glob, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import severe as S
for label, pat in zip(sys.argv[1::2], sys.argv[2::2]):
    c = []
    for f in sorted(glob.glob(pat, recursive=True)):
        mv = sorted(S.movable(f))
        if len(mv) != 2: continue
        ch = {}
        hdr = []
        for l in open(f):
            if l.startswith("_atom_site."): hdr.append(l.strip().split(".")[1])
            elif l.startswith(("ATOM", "HETATM")):
                r = l.split(); ix = {h: i for i, h in enumerate(hdr)}; el = r[ix["type_symbol"]].upper()
                if el == "H": continue
                ch.setdefault(r[ix["label_asym_id"]], []).append([float(r[ix[k]]) for k in ("Cartn_x", "Cartn_y", "Cartn_z")] + [S.R.get(el, 1.7)])
        a, b = np.array(ch[mv[0]]), np.array(ch[mv[1]])
        d = np.linalg.norm(a[:, None, :3] - b[None, :, :3], axis=-1)
        c.append(int((d < 0.75 * (a[:, None, 3] + b[None, :, 3])).sum()))
    c = np.array(c); print(f"{label}: n={len(c)} >=1 {np.mean(c>0):.2f} mean {c.mean():.2f} p90 {np.percentile(c,90):.0f} max {c.max()}", flush=True)

import sys, json, glob, os, numpy as np
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
pool, pb, out = sys.argv[1:4]
res = {}
for d in sorted(glob.glob(pool + '/*/')):
    i = os.path.basename(d.rstrip('/'))
    pdb = f'{pb}/{i}/{i}_protein.pdb'
    if not os.path.exists(pdb): continue
    P = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in open(pdb) if l.startswith(('ATOM', 'HETATM'))])
    rec = {}
    for f in sorted(glob.glob(d + 'rank*.sdf')):
        m = Chem.MolFromMolFile(f, sanitize=False)
        if m is None: rec[os.path.basename(f)] = None; continue
        X = m.GetConformer().GetPositions()
        dmin = float(np.sqrt(((X[:, None, :] - P[None, :, :]) ** 2).sum(-1)).min())
        ext = float(np.linalg.norm(X.max(0) - X.min(0)))
        rec[os.path.basename(f)] = [round(dmin, 2), round(ext, 2), bool(np.isfinite(X).all())]
    res[i] = rec
json.dump(res, open(out, 'w'))
bad = {i: {k: v for k, v in r.items() if v is None or v[0] > 15 or not v[2]} for i, r in res.items()}
bad = {i: v for i, v in bad.items() if v}
print(len(res), 'complexes;', len(bad), 'with a pose > 15 A from every receptor atom')
for i, v in bad.items(): print(i, v)

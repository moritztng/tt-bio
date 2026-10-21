import sys, numpy as np, torch
from pathlib import Path
from tt_bio._vendor.rfd4_prep.processing import prepare_cif
src, ref = sys.argv[1], sys.argv[2]
f, _ = prepare_cif(src, seed=0, device=torch.device("cpu"))
z = np.load(ref)
for k in z.files:
    a = f[k][0].detach().cpu(); a = (a.view(torch.int16) if a.dtype == torch.bfloat16 else a).numpy(); b = z[k]
    if not np.array_equal(a, b, equal_nan=True):
        d = np.abs(a.astype(np.float64) - b.astype(np.float64)); m = np.isfinite(d)
        print(k, a.dtype, a.shape, "n_diff", int((a != b).sum()), "maxabs", float(d[m].max()) if m.any() else None,
              "max|ref|", float(np.nanmax(np.abs(b))), "sign flips", int((np.sign(a) != np.sign(b)).sum()))

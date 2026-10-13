#!/usr/bin/env python3
"""Row statistics of the pair entering every K-th TriangleMultiplication call during a real fold, to size the
error of folding the input layer norm into the in-projection (rstd * (x W' - mu colsum W')): that form multiplies
raw x, so a reduced-fidelity matmul's error scales with rms(x) = sqrt(mu^2 + var), not with std. Prints quantiles
of |mu|/std and rms/std over valid rows (std > 1e-6), per sampled call and overall, as JSON lines on stdout.

    python perf/spd_trifuse/zstats.py --every 20 -- <perf/spd_gap_esmf/ab.py args>
"""
import json, runpy, sys
from pathlib import Path

argv = sys.argv[1:]
every = 20
if argv[:1] == ["--every"]:
    every, argv = int(argv[1]), argv[2:]
if argv[:1] == ["--"]:
    argv = argv[1:]
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

Q = torch.tensor([0.5, 0.9, 0.99, 1.0], dtype=torch.float64)
seen = {"n": 0}
pooled = {"mu_std": [], "rms_std": []}
orig = T.TriangleMultiplication.__call__


def probe(self, x, *args, **kw):
    seen["n"] += 1
    if seen["n"] % every == 1:
        z = ttnn.to_torch(x).double().reshape(-1, x.shape[-1])
        mu, sd = z.mean(-1), z.std(-1, unbiased=False)
        ok = sd > 1e-6
        r1 = (mu[ok].abs() / sd[ok])
        r2 = (z[ok].pow(2).mean(-1).sqrt() / sd[ok])
        pooled["mu_std"].append(r1.float()); pooled["rms_std"].append(r2.float())
        print(json.dumps({"call": seen["n"], "ending": bool(getattr(self, "ending", False)), "shape": list(x.shape),
                          "rows": int(ok.sum()), "mu_std_q": [round(v, 3) for v in torch.quantile(r1, Q).tolist()],
                          "rms_std_q": [round(v, 3) for v in torch.quantile(r2, Q).tolist()]}), flush=True)
    return orig(self, x, *args, **kw)


T.TriangleMultiplication.__call__ = probe
sys.argv = [str(root / "perf/spd_gap_esmf/ab.py")] + argv
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    for k, v in pooled.items():
        if v:
            a = torch.cat(v).double()
            a = a[torch.randperm(a.numel())[:1 << 24]]
            print(json.dumps({"pooled": k, "calls": len(v), "q50_90_99_max": [round(t, 3) for t in torch.quantile(a, Q).tolist()]}), flush=True)

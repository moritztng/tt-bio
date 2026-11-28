"""Card-free: host seconds per step the two exact duplicates cost at the gated ligand's row counts.

5SB2_1K2 on 5SAK_ZRY, 10 poses: 99,008 cross rows, 79,200 receptor rows (perf/recinv.json).
Random vectors and a randomly initialised model of the released config: the cost is shape-bound.
"""
import json
import sys
import time

import torch

from tt_bio.diffdock.harmonics import reversed_harmonics, spherical_harmonics

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
g = torch.Generator().manual_seed(0)
cvec = torch.randn(99008, 3, generator=g) * 10
rvec = torch.randn(79200, 3, generator=g) * 10


def best(fn, n=20):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts), sorted(ts)[len(ts) // 2]


sh = spherical_harmonics(cvec)
assert torch.equal(reversed_harmonics(sh), spherical_harmonics(-cvec))
out = {"threads": torch.get_num_threads(),
       "rev_recompute_s": best(lambda: spherical_harmonics(-cvec)),
       "rev_parity_s": best(lambda: reversed_harmonics(sh)),
       "rec_harmonics_s": best(lambda: spherical_harmonics(rvec))}
try:
    from tt_bio.diffdock import network
    from tt_bio.diffdock.job import DockOptions
    model = network.load(DockOptions(device="cpu"))
    m = getattr(model, "host", model).model
    rec_x = None
    with torch.no_grad():
        out["rec_edge_embedding_s"] = best(
            lambda: m.rec_edge_embedding(m.rec_distance_expansion(rvec.norm(dim=-1))))
except Exception as e:  # the released weights are not on every box
    out["rec_edge_embedding_s"] = f"skipped: {type(e).__name__}: {e}"
print(json.dumps(out, indent=1))

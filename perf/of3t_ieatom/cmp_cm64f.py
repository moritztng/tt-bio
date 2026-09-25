#!/usr/bin/env python3
"""of3t-ieatom: CM64F (59646c2bb + the two D264 commits, no D263) against PW64F, tensor by tensor.

    cmp_cm64f.py -> perf/of3t_ieatom/CMP_PW64F_CM64F.json

The compose is owed bit-identical: every named gradient, the name sets, and the loss. id()-keyed
`._wc.` aliases are excluded as in pwaslice compare.py (their names differ per process). If a
tensor differs, the first one in the order devstep dumped it is named.
"""
import json
from pathlib import Path

import torch

P = Path("/home/ttuser/of3t_pwaslice/grad_PW64F.pt")
C = Path("/home/ttuser/of3t_ieatom/grad_CM64F.pt")
H = Path(__file__).resolve().parent


def keep(d):
    return {k: v for k, v in d.items() if "._wc." not in k}


p, c = keep(torch.load(P, map_location="cpu")), keep(torch.load(C, map_location="cpu"))
common = [k for k in c if k in p]
differ = [k for k in common if not (p[k].shape == c[k].shape and p[k].dtype == c[k].dtype
                                    and torch.equal(p[k], c[k]))]
lp = json.loads((H.parent / "of3t_pwaslice/DEV_PW64F.json").read_text())
lc = json.loads((H / "DEV_CM64F.json").read_text())
prov = lambda d: f"{d['provenance']['git_commit'][:9]}, card {d['provenance']['card']}"
rec = {"pw64f": f"{P} ({prov(lp)})", "cm64f": f"{C} ({prov(lc)})",
       "wc_aliases_excluded": True, "n_pw64f": len(p), "n_cm64f": len(c), "common": len(common),
       "only_pw64f": sorted(set(p) - set(c)), "only_cm64f": sorted(set(c) - set(p)),
       "bit_identical": len(common) - len(differ), "differ_count": len(differ),
       "first_differing": differ[0] if differ else None, "differ": differ[:50],
       "loss_pw64f": lp["loss"], "loss_cm64f": lc["loss"],
       "loss_bit_identical": lp["loss"] == lc["loss"]}
(H / "CMP_PW64F_CM64F.json").write_text(json.dumps(rec, indent=1))
print(json.dumps({k: v for k, v in rec.items() if k != "differ"}, indent=1)[:3000])

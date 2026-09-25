#!/usr/bin/env python3
"""of3t-ieatom: PF64F (59646c2bb + D264 + D263) against PW64F, tensor by tensor.

    cmp_pf64f.py -> perf/of3t_ieatom/CMP_PW64F_PF64F.json

s_input reaches the trunk with the host leg's bytes (straight_through), so every gradient PW64F
carries must be bit-identical, the loss too. D263 only ADDS leaves: the input atom encoder's,
each owed non-zero. A PW64F name absent from PF64F must reappear under a new name with identical
bytes (AtomPairUpdate's extraction renamed the diffusion pair-update weights), or it is a loss.
"""
import json
from pathlib import Path

import torch

P = Path("/home/ttuser/of3t_pwaslice/grad_PW64F.pt")
F = Path("/home/ttuser/of3t_ieatom/grad_PF64F.pt")
H = Path(__file__).resolve().parent


def keep(d):
    return {k: v for k, v in d.items() if "._wc." not in k}


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


p, f = keep(torch.load(P, map_location="cpu")), keep(torch.load(F, map_location="cpu"))
common = [k for k in f if k in p]
differ = [k for k in common if not same(p[k], f[k])]
new = sorted(set(f) - set(p))
renamed = {k: next((n for n in new if same(p[k], f[n])), None) for k in sorted(set(p) - set(f))}
added = [n for n in new if n not in renamed.values()]
lp = json.loads((H.parent / "of3t_pwaslice/DEV_PW64F.json").read_text())
lf = json.loads((H / "DEV_PF64F.json").read_text())
prov = lambda d: f"{d['provenance']['git_commit'][:9]}, card {d['provenance']['card']}"
rec = {"pw64f": f"{P} ({prov(lp)})", "pf64f": f"{F} ({prov(lf)})", "wc_aliases_excluded": True,
       "common": len(common), "bit_identical": len(common) - len(differ),
       "differ_count": len(differ), "first_differing": differ[0] if differ else None,
       "differ": differ[:50], "renamed_bit_identical": renamed,
       "renamed_unmatched": [k for k, v in renamed.items() if v is None],
       "new_leaves": len(added), "new_nonzero": sum(bool(f[n].abs().sum() > 0) for n in added),
       "new_zero": [n for n in added if not f[n].abs().sum() > 0], "new": added,
       "loss_pw64f": lp["loss"], "loss_pf64f": lf["loss"],
       "loss_bit_identical": lp["loss"] == lf["loss"]}
(H / "CMP_PW64F_PF64F.json").write_text(json.dumps(rec, indent=1))
print(json.dumps({k: v for k, v in rec.items() if k not in ("differ", "new")}, indent=1))

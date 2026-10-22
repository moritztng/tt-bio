"""Is the upstream seam transplant a no-op because our values already match?"""
import sys, torch
from pathlib import Path
sys.path.insert(0, "/home/moritz/.coworker/wt/rfd4-cond-chem")
from tt_bio import rfd4
from tt_bio.rfd4 import engine as E

cif, seam_pt, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
features, _ = rfd4.featurize(Path(cif), seed=seed)
ours = E.preprocess(features)
up = torch.load(seam_pt, weights_only=False)["batch"]
for key in ("chain_iid", "molecule_iid", "condition_InternalCoordinate_1_atom"):
    src = next((v for k, v in up.items() if k == key or k.endswith("." + key)), None)
    o = ours[key]
    s = src.to(o.dtype)
    same = o.shape == s.shape and torch.equal(o, s)
    print(f"{key}: ours{tuple(o.shape)} upstream{tuple(s.shape)} identical={same}", end="")
    if not same and o.shape == s.shape:
        d = (o.float() - s.float()).abs()
        print(f" max_abs_diff={d.max().item():.6g} n_differing={(d>0).sum().item()}/{d.numel()}")
    else:
        print()

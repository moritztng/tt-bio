#!/usr/bin/env python3
"""D174's FORWARD effect, which is the part users get today.

The gradient arms already save our stack output (`dev_grad.py` writes `s` and `z` beside the
grads), and `block47_boundary.pt` carries upstream's own float64 output at the same boundary.
So the forward A/B costs no card time: it is a comparison of tensors already on disk.

Three questions, each answered on the REAL block and on the padded region separately, because
the whole of D174 is that those two regions behave differently:

  1. does masking change the real block at all? `of3t-auxfind` measured EXACTLY 0.0 on upstream
     (arm P); if our port agrees, the mask is inference-neutral where it counts and the release
     question is only about time.
  2. does it change the padded region? It must, or the lever did nothing.
  3. does it move our forward toward upstream's -- the D19 question (7.811e-03 per pairformer
     block composing to 2.792e-01 over 48)?
"""
import json
import sys
import torch

O = "/tmp/of3t/of3t-ditmodel"
B = torch.load("/home/ttuser/of3t_trunk043ref/boundary_n384.pt", map_location="cpu",
               weights_only=False)
C = torch.load("/home/ttuser/of3t_gradients/cap/block47_boundary.pt", map_location="cpu",
               weights_only=False)
sm = B["single_mask"].reshape(-1)
real1 = sm > 0
real2 = (sm[:, None] * sm[None, :]) > 0
ref_s, ref_z = C["out"][0].to(torch.float64), C["out"][1].to(torch.float64)


def n(t):
    return float(torch.linalg.vector_norm(t))


def rel(a, b):
    d = n(a - b)
    return d / n(b) if n(b) else None


arms = {}
for tag in ("MASKOFF", "MASKON", "MASKONES"):
    try:
        d = torch.load(f"{O}/dev_{tag}_n384.pt", map_location="cpu", weights_only=False)
    except FileNotFoundError:
        continue
    arms[tag] = (d["s"].to(torch.float64), d["z"].to(torch.float64))

out = {"what": __doc__.strip().splitlines()[0],
       "tokens": int(sm.numel()), "real_tokens": int(real1.sum()),
       "pad_tokens": int((~real1).sum()),
       "reference": {"path": "/home/ttuser/of3t_gradients/cap/block47_boundary.pt",
                     "dtype": "float64", "s_norm": n(ref_s), "z_norm": n(ref_z),
                     "s_norm_real": n(ref_s[0][real1]), "z_norm_real": n(ref_z[0][real2]),
                     "s_norm_pad": n(ref_s[0][~real1]), "z_norm_pad": n(ref_z[0][~real2])},
       "vs_upstream_float64": {}, "arm_vs_arm": {}}

for tag, (s, z) in arms.items():
    out["vs_upstream_float64"][tag] = {
        "s_rel_real": rel(s[0][real1], ref_s[0][real1]),
        "z_rel_real": rel(z[0][real2], ref_z[0][real2]),
        "s_rel_pad": rel(s[0][~real1], ref_s[0][~real1]),
        "z_rel_pad": rel(z[0][~real2], ref_z[0][~real2]),
        "s_norm_pad": n(s[0][~real1]), "z_norm_pad": n(z[0][~real2]),
    }

base = arms.get("MASKOFF")
if base is not None:
    for tag, (s, z) in arms.items():
        if tag == "MASKOFF":
            continue
        bs, bz = base
        out["arm_vs_arm"][f"{tag}_minus_MASKOFF"] = {
            "s_bit_identical": bool(torch.equal(s, bs)),
            "z_bit_identical": bool(torch.equal(z, bz)),
            "s_absdiff_real": n(s[0][real1] - bs[0][real1]),
            "z_absdiff_real": n(z[0][real2] - bz[0][real2]),
            "s_rel_real": rel(s[0][real1], bs[0][real1]),
            "z_rel_real": rel(z[0][real2], bz[0][real2]),
            "s_absdiff_pad": n(s[0][~real1] - bs[0][~real1]),
            "z_absdiff_pad": n(z[0][~real2] - bz[0][~real2]),
        }

json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))

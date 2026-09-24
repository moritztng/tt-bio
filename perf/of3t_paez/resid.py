#!/usr/bin/env python3
"""of3t-paez: the shape of each side's pred_xyz residual against float64's (PRED_SPLIT.preds.pt).

d = pred_side - pred_f64 on real tokens, regressed on the noise the denoise left behind,
u = x_noisy - pred_f64 (x_noisy at the same token gather), and on the update itself,
v = pred_f64 - x_noisy * c_skip. Reports alpha = <d,u>/<u,u>, the fraction of |d|^2 along u,
and the non-rigid residual left after a Kabsch fit of pred_side onto pred_f64.
"""
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "of3t_fullstep64"))
import ref_step  # noqa: E402
from tt_bio.train.openfold3 import denoise_draw  # noqa: E402

bm = ref_step.bm
BATCH = Path("/home/ttuser/of3t-campaign-refs/bundle_min_043/batch_step003.pt")
preds = torch.load(sys.argv[1], weights_only=False)
batch = bm.move(torch.load(BATCH, weights_only=False), "cpu", torch.float64)
n_atom = int(batch["ground_truth"]["atom_positions"].shape[-2])
sigma, eps = denoise_draw(20260922, n_atom)
amask = batch["atom_mask"][0]
xl_true = batch["ground_truth"]["atom_positions"][0] * amask[:, None]
xl_noisy = (xl_true + sigma * torch.as_tensor(eps, dtype=torch.float64)) * amask[:, None]
tok = batch["token_mask"][0]
real = torch.nonzero(tok > 0, as_tuple=True)[0]
rep = batch["start_atom_index"][0].long()[real]
xn = xl_noisy[rep]
xt = xl_true[rep]
sd = 16.0
c_skip = sd * sd / (sd * sd + sigma * sigma)
p0 = preds["f64"][real]
u = xn - p0
v = p0 - c_skip * xn


def kabsch_resid(a, b):
    a0, b0 = a - a.mean(0), b - b.mean(0)
    U, S, Vt = torch.linalg.svd(a0.T @ b0)
    dd = torch.sign(torch.det(U @ Vt))
    D = torch.diag(torch.tensor([1.0, 1.0, float(dd)], dtype=a.dtype))
    R = U @ D @ Vt
    return float((a0 @ R - b0).norm() / b0.norm())


rec = {"sigma": sigma, "c_skip": c_skip, "real_tokens": int(real.numel()),
       "rms_noise_left_A": float(u.pow(2).sum(-1).mean().sqrt()),
       "rms_f64_to_truth_A": float((p0 - xt).pow(2).sum(-1).mean().sqrt()), "sides": {}}
for k, p in preds.items():
    if k == "f64":
        continue
    d = p[real] - p0
    a_u = float((d * u).sum() / (u * u).sum())
    a_v = float((d * v).sum() / (v * v).sum())
    rec["sides"][k] = {
        "rms_resid_A": float(d.pow(2).sum(-1).mean().sqrt()),
        "alpha_noise_left": a_u, "frac_sq_along_noise_left": a_u ** 2 * float((u * u).sum() / (d * d).sum()),
        "alpha_update": a_v, "frac_sq_along_update": a_v ** 2 * float((v * v).sum() / (d * d).sum()),
        "rms_to_truth_A": float((p[real] - xt).pow(2).sum(-1).mean().sqrt()),
        "nonrigid_rel_after_kabsch": kabsch_resid(p[real], p0),
        "raw_rel": float(d.norm() / p0.norm()),
    }
print(json.dumps(rec, indent=1))
json.dump(rec, open(sys.argv[2], "w"), indent=1)

#!/usr/bin/env python3
"""of3t-paez: how much pae gradient a pred_xyz perturbation of a given size carries by chance.

    label_null.py --cache RATIO.cache.pt --preds PRED_SPLIT.preds.pt --out LABEL_NULL.json

The pae label is a step function of pred_xyz, so the gradient one perturbation carries through
the labels is one draw from a distribution. Drawn here at float64's zij_conf (the head and loss
exactly as pred_split.py), against float64's labels:
  iid   pred_f64 + N(0, r^2) per coordinate on real tokens, r = the device's and bf16's residual
        RMS (RESID.json), 24 seeds each;
  shape pred_f64 + lam * (pred_side - pred_f64) for lam in {-1, 0.5, 1, 2}, each side's own
        residual shape, so a sign flip and a rescale of the same error are read too.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "of3t_fullstep64"))
import ref_step  # noqa: E402
import score  # noqa: E402

CK = Path("/home/ttuser/of3-weights/of3-p2-155k.pt")
BATCH = Path("/home/ttuser/of3t-campaign-refs/bundle_min_043/batch_step003.pt")
P = "aux_heads.pae."
W_PAE = 1e-4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(8)
    from tt_bio.train import losses
    zc = torch.load(a.cache, weights_only=False, mmap=True)["f64"]["zij_conf"].clone()
    preds = torch.load(a.preds, weights_only=False)
    model = ref_step.load(torch.float64, CK, 20260919, None)[1]
    lab = ref_step.labels(BATCH)
    ah = model.aux_heads
    params = {n: p for n, p in model.named_parameters() if n.startswith(P)}
    with torch.enable_grad():
        logits = ah.pae(zc)

    def grad(px):
        _v, g = losses.pae(logits.detach().numpy(), px.numpy(), lab["true_xyz"],
                           lab["coord_mask"], lab["frame_atom_index"])
        gs = torch.autograd.grad(logits, list(params.values()),
                                 torch.from_numpy(W_PAE * np.asarray(g)).double(), retain_graph=True)
        return dict(zip(params, gs))

    p0 = preds["f64"]
    g0 = grad(p0)
    rel = lambda g: score.triple([(g0[n], g[n]) for n in sorted(g0)])["rel"]
    real = torch.from_numpy(np.asarray(lab["coord_mask"], bool))
    resid = json.load(open(a.preds.parent / "RESID.json"))["sides"]
    rec = {"iid": {}, "shape": {}}
    for side in ("dev", "bf16"):
        r = resid[side]["rms_resid_A"]
        gen = torch.Generator().manual_seed(20260924)
        vals = []
        for _s in range(a.seeds):
            n = torch.randn(p0.shape, generator=gen, dtype=torch.float64) * (r / 3 ** 0.5)
            vals.append(rel(grad(p0 + n * real[:, None])))
        v = np.array(vals)
        rec["iid"][side] = {"rms_A": r, "n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                            "min": float(v.min()), "max": float(v.max()), "values": v.tolist()}
        d = preds[side] - p0
        rec["shape"][side] = {str(l): rel(grad(p0 + l * d)) for l in (-1.0, 0.5, 1.0, 2.0)}
        print(side, json.dumps({k: rec["iid"][side][k] for k in ("rms_A", "mean", "sd", "min", "max")}),
              rec["shape"][side], flush=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""of3t-paez step 3: which side carries the pae labels' `pred_xyz`, 384 wide.

    pred_split.py --cache RATIO.cache.pt --conf C.pt --outputs O.pt --out PRED_SPLIT.json

The pae (and pde) label is binned from the denoise arm's `pred_xyz`; GRAD_SPLIT.json shows the
device's labels carry D267. Upstream's denoise arm (`ref_step.denoise_arm`, same draw as the
scored step) is run at: float64's trunk outputs in float64 (self-control against grad_split's
pred), upstream bf16's trunk outputs under its own bf16 recipe (= the bar's own labels), and the
device's trunk outputs in float64 (= what the device's inputs carry into pred_xyz). Per side:
pred_xyz rel against float64, pae / pde label bins that differ from float64's, and the pae
gradient at float64's z with that side's labels, against the banked float64 gradient.
"""
from __future__ import annotations

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

bm = ref_step.bm
R = Path("/home/ttuser/of3t-campaign-refs")
REF = Path("/home/ttuser/of3t_confpfe/ref384c")
BATCH = R / "bundle_min_043/batch_step003.pt"
CK = Path("/home/ttuser/of3-weights/of3-p2-155k.pt")
SEED = 20260922
P = "aux_heads.pae."
W_PAE = 1e-4          # of3_loss_weights("initial_training", "weighted-pdb")["pae"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--conf", type=Path, required=True)
    ap.add_argument("--outputs", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    bm.pin_deterministic_kernels(True)
    from tt_bio.train import losses
    from tt_bio.train.openfold3 import denoise_draw
    c = torch.load(a.cache, weights_only=False)
    W = c["f64"]["z_trunk"].shape[-2]
    conf = torch.load(a.conf, weights_only=False)["inputs"]
    dev_in = {"s_input": conf["s_input"].double().reshape(1, -1, conf["s_input"].shape[-1])[:, :W],
              "s_trunk": conf["s_trunk"].double().reshape(1, -1, conf["s_trunk"].shape[-1])[:, :W],
              "z_trunk": conf["z_trunk"].double().reshape(1, *conf["z_trunk"].shape[-3:])[:, :W, :W]}
    lab = ref_step.labels(BATCH)
    preds = {"dev": torch.load(a.outputs, weights_only=False)["pred_xyz"].reshape(-1, 3)[:W]}

    def arm(dt, policy, src):
        model = ref_step.load(dt, CK, 20260919, None)[1]
        batch = bm.move(torch.load(BATCH, weights_only=False), "cpu", dt)
        draw = denoise_draw(SEED, int(batch["ground_truth"]["atom_positions"].shape[-2]))
        with torch.no_grad(), bm.cast_policy(policy, "cpu"):
            o = ref_step.denoise_arm(model, batch, src["s_input"].to(dt), src["s_trunk"].to(dt),
                                     src["z_trunk"].to(dt), draw)
        return model, o["pred_xyz"].double()

    model, preds["f64"] = arm(torch.float64, "removed", c["f64"])
    _m, preds["f64_at_dev_inputs"] = arm(torch.float64, "removed", dev_in)
    _m, preds["f64_at_bf16_inputs"] = arm(torch.float64, "removed", c["bf"])
    del _m
    _m, preds["bf16"] = arm(torch.float32, "bf16", c["bf"])
    del _m

    fai = np.asarray(lab["frame_atom_index"])
    m = np.asarray(lab["coord_mask"], bool)
    tf = np.asarray(lab["true_xyz"], np.float64)
    pm = (m[fai].sum(-1) >= 3)[:, None] & m[None, :]
    pdm = m[:, None] & m[None, :]

    def bins(px):
        px = px.numpy()
        sq = ((losses.express_in_frame(px, px[..., fai, :])
               - losses.express_in_frame(tf, tf[..., fai, :])) ** 2).sum(-1)
        pae = np.where(pm, losses.bin_index(sq * pm, *losses.PAE_GRID, n_edges=65, square=True), -1)
        err = np.abs(losses._pdist(px) - losses._pdist(tf))
        pde = np.where(pdm, losses.bin_index(err, *losses.PDE_GRID, n_edges=65), -1)
        return pae, pde

    # pae gradient at float64's zij_conf with each side's labels: losses.pae's logit gradient
    # (the objective's own pae term) back through upstream's float64 head. Scored against the
    # same construction at float64's labels, which grad_split showed equals the banked gradient.
    ah = model.aux_heads
    params = {n: p for n, p in model.named_parameters() if n.startswith(P)}
    zc = c["f64"]["zij_conf"]

    def grad(px):
        with torch.enable_grad():
            logits = ah.pae(zc)
            v, g = losses.pae(logits.detach().numpy(), px.numpy(), lab["true_xyz"],
                              lab["coord_mask"], lab["frame_atom_index"])
            gs = torch.autograd.grad(logits, list(params.values()),
                                     torch.from_numpy(W_PAE * np.asarray(g)).double())
        return dict(zip(params, gs)), float(v)

    g0, v0 = grad(preds["f64"])
    banked = {k: v.double() for k, v in torch.load(REF / "f64/grads_f64.pt", weights_only=False).items()
              if k.startswith(P)}
    b0 = bins(preds["f64"])
    rec = {"width": W, "n_pae_pairs": int(pm.sum()), "n_pde_pairs": int(pdm.sum()),
           "self_control_grad_vs_banked": score.triple([(banked[n], g0[n]) for n in sorted(g0)]),
           "self_control_pae_value": v0, "sides": {}}
    for k, px in preds.items():
        b = bins(px)
        r = {"pred_xyz_rel": float((px - preds["f64"]).norm() / preds["f64"].norm()),
             "pred_xyz_max_abs_A": float((px - preds["f64"]).abs().max()),
             "pae_bins_differ": int((b[0] != b0[0]).sum()),
             "pde_bins_differ": int((b[1] != b0[1]).sum())}
        g, v = grad(px)
        r["pae_grad_at_f64_z"] = score.triple([(g0[n], g[n]) for n in sorted(g0)])
        r["pae_value_at_f64_z"] = v
        rec["sides"][k] = r
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

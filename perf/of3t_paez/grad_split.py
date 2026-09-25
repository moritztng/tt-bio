#!/usr/bin/env python3
"""of3t-paez step 2: the `aux_heads.pae` gradient split at the head's inputs, 384 wide.

    grad_split.py --conf C.pt --outputs O.pt --dev-grad G.pt --out GRAD_SPLIT.json

Upstream 0.4.3's float64 pae head (LayerNorm + Linear) and our objective in float64, backward to
the head's four parameters, evaluated at chosen inputs: the head input `zij_conf` and the
`pred_xyz` the pae label is binned from (both are what the pae gradient reads; the seed is
w * (softmax(logits) - onehot(label)) on masked pairs). Sides: f64 = upstream float64's own step
(trunk, denoise arm and confidence Pairformer at ref384c's rolled-out structure); dev = the
CF384 tree's own forward (devstep_out.py / devstep.py --conf-dump). Each mixed evaluation is
scored against ref384c's banked float64 gradient (`carried`) and the device's banked gradient
against the float64 head at the device's own inputs (`head backward`).
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
BATCH, BATCH_SHA = R / "bundle_min_043/batch_step003.pt", \
    "3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f"
CK = Path("/home/ttuser/of3-weights/of3-p2-155k.pt")
SEED = 20260922
P = "aux_heads.pae."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=Path, required=True)
    ap.add_argument("--outputs", type=Path, required=True)
    ap.add_argument("--dev-grad", type=Path, required=True)
    ap.add_argument("--bijection", type=Path, required=True)
    ap.add_argument("--shapes", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if ref_step.sha256_file(BATCH) != BATCH_SHA:
        raise SystemExit("batch sha256 mismatch")
    bm.pin_deterministic_kernels(True)
    dt = torch.float64
    cfg, model, _d, _ck = ref_step.load(dt, CK, 20260919, 32)      # run_ref.sh's chunk size
    lab = ref_step.labels(BATCH)
    batch = bm.move(torch.load(BATCH, weights_only=False), "cpu", dt)
    from tt_bio.train import losses
    from tt_bio.train.openfold3 import denoise_draw
    n_atom = int(batch["ground_truth"]["atom_positions"].shape[-2])
    draw = denoise_draw(SEED, n_atom)
    lab["edm_scale"] = float(losses.edm_scale(draw[0], float(model.diffusion_module.sigma_data)))
    ref_f64 = json.load(open(REF / "f64/REF_F64.json"))

    ah = model.aux_heads
    seen = {}
    h = ah.pae.register_forward_pre_hook(lambda _m, args: seen.__setitem__("zij", args[0].detach()))
    rx = torch.load(REF / "f64/rollout_f64.pt", weights_only=False)["repr_x"]
    with torch.no_grad(), bm.cast_policy("removed", "cpu"):
        out_f64, _xl, _rx, _rec = ref_step.trunk_and_heads(model, batch, repr_x=rx.to(dt), cfg=cfg,
                                                           seed=SEED, draw=draw)
    h.remove()
    loss_f64, _b, _s = ref_step.objective(lab, out_f64)
    W = batch["token_mask"].shape[-1]
    zij = {"f64": seen["zij"]}
    conf = torch.load(a.conf, weights_only=False)
    z = conf["outputs"]["zij_conf"].double()
    zij["dev"] = z.reshape(1, *z.shape[-3:])[:, :W, :W]
    dev_out = torch.load(a.outputs, weights_only=False)
    pred = {"f64": out_f64["pred_xyz"].detach(), "dev": dev_out["pred_xyz"].reshape(-1, 3)[:W]}

    params = {n: p for n, p in model.named_parameters() if n.startswith(P)}

    def grad(zk, pk):
        with torch.enable_grad():
            logits = ah.pae(zij[zk])
            o = {k: v.detach() for k, v in out_f64.items()}
            o["pae_logits"], o["pred_xyz"] = logits.detach(), pred[pk]
            o["pred_dist"] = ref_step.safe_pdist(pred[pk])
            _l, bd, seeds = ref_step.objective(lab, o)
            gs = torch.autograd.grad(logits, list(params.values()),
                                     torch.from_numpy(np.asarray(seeds["pae_logits"])).to(dt))
        return dict(zip(params, gs)), bd["pae"]["value"]

    ref = {k: v.double() for k, v in torch.load(REF / "f64/grads_f64.pt", weights_only=False).items()
           if k.startswith(P)}
    bf = {k: v.double() for k, v in torch.load(REF / "bf16/grads_bf16.pt", weights_only=False).items()
          if k.startswith(P)}
    shapes = json.loads(a.shapes.read_text())["shapes"]
    dev_g, _m = score.to_upstream(score.load_device(str(a.dev_grad), shapes),
                                  json.loads(a.bijection.read_text()))
    dev = {k: v for k, v in dev_g.items() if k.startswith(P)}

    def tri(x, y):   # rel of y against x, over the section and per tensor
        return {"section": score.triple([(x[k], y[k]) for k in sorted(ref)]),
                "per_tensor": {k: score.triple([(x[k], y[k])]) for k in sorted(ref)}}

    g = {}
    pae_val = {}
    for zk in ("f64", "dev"):
        for pk in ("f64", "dev"):
            g[f"z_{zk}__label_{pk}"], pae_val[f"z_{zk}__label_{pk}"] = grad(zk, pk)
    rec = {"batch": BATCH.name, "width": W, "conf_dump": str(a.conf), "outputs": str(a.outputs),
           "dev_grad": str(a.dev_grad),
           "self_control": {"loss": loss_f64, "ref_loss": ref_f64["loss"],
                            "loss_abs_diff": abs(loss_f64 - ref_f64["loss"]),
                            "grad_vs_banked": tri(ref, g["z_f64__label_f64"])["section"]},
           "pae_loss_value": pae_val, "ref_pae_value": ref_f64["breakdown"]["pae"]["value"],
           "banked": {"dev_vs_f64": tri(ref, dev), "bf16_vs_f64": tri(ref, bf)},
           "carried": {k: tri(ref, v) for k, v in g.items()},
           "head_backward": tri(g["z_dev__label_dev"], dev)}
    # How many masked pairs change pae bin between the two pred_xyz labellings.
    fai = np.asarray(lab["frame_atom_index"])
    m = np.asarray(lab["coord_mask"], bool)
    tf = np.asarray(lab["true_xyz"], np.float64)
    bins = {}
    for pk, px in pred.items():
        px = px.numpy()
        sq = ((losses.express_in_frame(px, px[..., fai, :])
               - losses.express_in_frame(tf, tf[..., fai, :])) ** 2).sum(-1)
        pm = (m[fai].sum(-1) >= 3)[:, None] & m[None, :]
        bins[pk] = np.where(pm, losses.bin_index(sq * pm, *losses.PAE_GRID, n_edges=65, square=True), -1)
    rec["label_bins"] = {"n_masked_pairs": int((bins["f64"] >= 0).sum()),
                         "n_differ": int((bins["f64"] != bins["dev"]).sum()),
                         "mean_abs_bin_shift": float(np.abs(bins["f64"] - bins["dev"])[bins["f64"] >= 0].mean())}
    rec["pred_xyz_rel"] = float((pred["dev"] - pred["f64"]).norm() / pred["f64"].norm())
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("self_control", "pae_loss_value", "ref_pae_value",
                                          "label_bins", "pred_xyz_rel")}, indent=1))
    for grp in ("banked", "carried"):
        for k, v in rec[grp].items():
            s = v["section"]
            print(f"{grp:8s} {k:22s} rel {s['rel']:.4e} r {s['r']:.4f} cos {s['cos']:.6f}")
    s = rec["head_backward"]["section"]
    print(f"head_backward rel {s['rel']:.4e} r {s['r']:.4f} cos {s['cos']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score tt-bio's PXDesign-d against the upstream torch reference, component by component.

Both sides consume the same captured PD-L1 input dict, so nothing here depends on a
featurizer. `upstream_ref.py` produces the reference tensors on CPU; this script runs the
device port on the same inputs and reports the delta.

    # 1. reference (CPU, no device)
    ~/protenix_ref_venv/bin/python scripts/pxdesign_port/upstream_ref.py \
        --stage cond --out /tmp/ref_cond.pt
    # 2. port (device)
    TT_VISIBLE_DEVICES=0 python3 scripts/pxdesign_port/upstream_parity.py \
        --stage cond --ref /tmp/ref_cond.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def report(name, got, want):
    import torch
    got, want = got.detach().float(), want.detach().float()
    assert got.shape == want.shape, f"{name}: {tuple(got.shape)} vs {tuple(want.shape)}"
    d = (got - want).abs()
    denom = want.abs().mean().clamp_min(1e-12)
    g, w = got.reshape(-1).double(), want.reshape(-1).double()
    gc, wc = g - g.mean(), w - w.mean()
    pcc = float((gc @ wc) / (gc.norm() * wc.norm()).clamp_min(1e-30))
    rec = {"key": name, "shape": list(got.shape), "max_abs": float(d.max()),
           "mean_abs": float(d.mean()), "rel_mean": float(d.mean() / denom), "pcc": pcc,
           "bit_exact": bool(torch.equal(got, want))}
    print(f"[parity] {name:<24} max|d|={rec['max_abs']:.4e} mean|d|={rec['mean_abs']:.4e} "
          f"rel={rec['rel_mean']:.3e} PCC={pcc:.8f} exact={rec['bit_exact']}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("cond", "denoise"))
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from design_e2e import load_design_inputs
    ref = torch.load(args.ref, weights_only=False)
    feats = load_design_inputs()
    NT = int(feats["atom_to_token_idx"].max()) + 1

    from tt_bio.pxdesign.model import ProtenixDesign
    import os
    model = ProtenixDesign.load_from_checkpoint(
        str(Path("~/pxdesign_release_data/checkpoint/pxdesign_v0.1.0.pt").expanduser()))
    out = []

    # z_trunk is a host gather, so it is comparable before anything touches the device.
    out.append(report("z_trunk", model._condition_z(feats, NT), ref["z_trunk"]))

    cond, aux = model._trunk_cond(feats)
    out.append(report("s_inputs", torch.as_tensor(cond["s_inputs"]), ref["s_inputs"]))
    out.append(report("s_trunk", torch.as_tensor(cond["s_trunk"]), ref["s_trunk"]))

    if args.stage == "denoise":
        for k, d in sorted(ref["denoise"].items()):
            got = model.diffusion.denoise(
                d["x_noisy"], torch.tensor([d["sigma"]], dtype=torch.float32), cond)
            out.append(report(f"denoised@step{k}(sigma={d['sigma']:.4g})",
                              got, d["denoised"]))

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"[parity] -> {args.out}")


if __name__ == "__main__":
    main()

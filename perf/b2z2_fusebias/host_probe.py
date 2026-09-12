#!/usr/bin/env python3
"""How big is the perturbation the fused bias stacks actually inject? CPU only, real weights.

`_fuse_bias_stack` is host torch, not a kernel: `F.layer_norm` + one wide `F.linear` against a
per-layer LayerNorm+Linear loop. So the arithmetic difference is the same on every box we own, and
the only thing an architecture can change is what the device does with the perturbed conditioning
downstream. This measures the input half of that sentence on Boltz-2's own weights, at the three
call sites `DiffusionConditioning` has, so the fold result can be read against a number rather
than against the docstring's claim.

    host_probe.py --ckpt ~/.boltz/boltz2_conf.ckpt --rows 4096 --out <json>
"""
import argparse
import collections
import json
from pathlib import Path

import torch
import torch.nn.functional as F

STACKS = ("atom_enc_proj_z", "atom_dec_proj_z", "token_trans_proj_z")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path.home() / ".boltz" / "boltz2_conf.ckpt")
    ap.add_argument("--rows", type=int, default=4096, help="rows of the production width to draw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    torch.manual_seed(a.seed)
    report = {"ckpt": str(a.ckpt), "rows": a.rows, "seed": a.seed, "stacks": {}}

    for stack in STACKS:
        pre = f"diffusion_conditioning.{stack}."
        idx = sorted({int(k[len(pre):].split(".")[0]) for k in sd if k.startswith(pre)})
        lay = collections.OrderedDict()
        for i in idx:
            lay[i] = {"nw": sd[f"{pre}{i}.0.weight"].float(), "nb": sd[f"{pre}{i}.0.bias"].float(),
                      "pw": sd[f"{pre}{i}.1.weight"].float()}
        c = lay[idx[0]]["nw"].numel()
        h = lay[idx[0]]["pw"].shape[0]
        x = torch.randn(a.rows, c)

        # shipped-off path: normalise, scale, project, once per layer, then concatenate
        per = torch.cat([F.linear(F.layer_norm(x, (c,), l["nw"], l["nb"], 1e-5), l["pw"])
                         for l in lay.values()], dim=-1)
        # shipped-on path: one affine-free LayerNorm, one wide Linear with a constant bias
        w = torch.cat([l["pw"] * l["nw"] for l in lay.values()], dim=0)
        b = torch.cat([l["pw"] @ l["nb"] for l in lay.values()], dim=0)
        fused = F.linear(F.layer_norm(x, (c,), eps=1e-5), w, b)

        d = (fused - per).abs()
        scale = per.abs()
        report["stacks"][stack] = {
            "layers": len(idx), "c": c, "h": h, "out_width": h * len(idx),
            "max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "rms_out": float(per.pow(2).mean().sqrt()),
            "max_rel_to_rms": float(d.max() / per.pow(2).mean().sqrt()),
            "mean_rel_to_rms": float(d.mean() / per.pow(2).mean().sqrt()),
            "max_rel_elementwise": float((d / scale.clamp_min(1e-6)).max()),
            "allclose_1e5": bool(torch.allclose(fused, per, rtol=1e-5, atol=1e-6)),
        }
        print(f"  {stack:20s} {len(idx):2d} layers  C={c:4d} H={h:3d}  "
              f"max {d.max():.3e}  mean {d.mean():.3e}  "
              f"max/rms {report['stacks'][stack]['max_rel_to_rms']:.3e}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

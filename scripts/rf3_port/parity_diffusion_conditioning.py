#!/usr/bin/env python3
"""Score RF3's diffusion conditioning on ttnn, against a ceiling measured in the same run.

The encoder pass had to learn this twice: a threshold is meaningless until the roof is
measured for the module under test. So this reports device-vs-reference AND the
reference's own bf16 disagreement (fp32 vs cpu-bf16 autocast) side by side, and prints
the ratio rather than a verdict.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
SEL = {"template": dict(template_selection=["9dfn_A"]), "cyclic": dict(cyclic_chains=["A"])}


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d else float("nan")


def rel_rms(a, b):
    return float((a - b).pow(2).mean().sqrt() / b.std())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fixture", default="rna")
    ap.add_argument("--capture", default="/home/ttuser/rf3_ref_work/fi/rna_dc.pt")
    args = ap.parse_args()

    import ttnn
    from tt_bio.rf3.diffusion_conditioning import DiffusionConditioning
    from tt_bio.rf3.feature_init import relpos_features
    from tt_bio.rf3.featurize import featurize
    from tt_bio.rf3.weights import load_reference
    from tt_bio.tenstorrent import get_device

    cap = torch.load(args.capture, weights_only=False)
    t, f_cap, s_inputs, s_trunk, z_trunk = cap["in"][:5]
    want_S, want_Z = cap["out"][0], cap["out"][1]

    net, _ = load_reference(args.ckpt, num_steps=2)
    dc_ref = net.diffusion_module.diffusion_conditioning

    d = REPO / "scripts/rf3_port/parity_artifacts" / args.fixture
    prev = os.getcwd(); os.chdir(d)
    try:
        out = featurize("input.json", n_recycles=2, diffusion_batch_size=1, seed=42,
                        **SEL.get(args.fixture, {}))[0]
    finally:
        os.chdir(prev)
    f = out["feats"]

    # ---- the reference's own bf16 ceiling, same inputs
    def ref(bf16):
        ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if bf16
               else torch.autocast("cpu", enabled=False))
        with torch.no_grad(), ctx:
            s, z = dc_ref(t, f, s_inputs, s_trunk, z_trunk)
        return s.float(), z.float()

    hi_s, hi_z = ref(False)
    lo_s, lo_z = ref(True)

    # ---- device
    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    def tt(x):
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)

    sd = {k[len("shadow.diffusion_module.diffusion_conditioning."):]: v.float()
          for k, v in torch.load(args.ckpt, map_location="cpu",
                                 weights_only=False)["model"].items()
          if k.startswith("shadow.diffusion_module.diffusion_conditioning.")}
    dc = DiffusionConditioning(sd, cfg, sigma_data=dc_ref.sigma_data)

    rp = relpos_features(f, r_max=dc_ref.relative_position_encoding.r_max,
                         s_max=dc_ref.relative_position_encoding.s_max)
    got_s, got_z = dc(tt(rp.unsqueeze(0)), tt(z_trunk.unsqueeze(0)),
                      tt(s_trunk.unsqueeze(0)), tt(s_inputs.unsqueeze(0)), t)

    def back(x):
        return torch.Tensor(ttnn.to_torch(x)).float()

    rows = []
    for name, got, want, hi, lo in (
        ("S_I", back(got_s), want_S, hi_s, lo_s),
        ("Z_II", back(got_z), want_Z, hi_z, lo_z),
    ):
        got = got.reshape(want.shape)
        ceil = rel_rms(lo.reshape(want.shape), hi.reshape(want.shape))
        dev_e = rel_rms(got, want)
        rows.append({"tensor": name, "shape": list(want.shape),
                     "pcc": round(pcc(got, want), 7),
                     "rel_rms": round(dev_e, 6),
                     "bf16_ceiling": round(ceil, 6),
                     "x_ceiling": round(dev_e / ceil, 2) if ceil else None})
    print(json.dumps({"fixture": args.fixture, "scores": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

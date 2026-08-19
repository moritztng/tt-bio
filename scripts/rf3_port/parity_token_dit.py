#!/usr/bin/env python3
"""Score RF3's 24-block token DiT, ceiling measured in the same run, both fp32_softmax arms."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
PREFIX = "shadow.diffusion_module.diffusion_transformer."


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
    ap.add_argument("--capture", default="/home/ttuser/rf3_ref_work/fi/rna_dit.pt")
    args = ap.parse_args()

    import ttnn
    from tt_bio.rf3.token_dit import TokenDiffusionTransformer
    from tt_bio.rf3.weights import load_reference
    from tt_bio.tenstorrent import get_device

    cap = torch.load(args.capture, weights_only=False)
    a_in, s_in, z_in = cap["in"][0], cap["in"][1], cap["in"][2]
    want = cap["out"]

    net, _ = load_reference(args.ckpt, num_steps=2)
    ref = net.diffusion_module.diffusion_transformer

    # This module pins its own precision: AttentionPairBiasDiffusion sets
    # force_bfloat16 = True and casts A_I to bf16 inside forward, so toggling autocast
    # alone leaves both arms in bf16 and "the ceiling" would come out as zero. The cast
    # sits AFTER the `Beta_II is not None` early return, so it applies to this token
    # stack and NOT to the atom stacks, which is why they measured a ceiling normally.
    def run(bf16):
        for blk in ref.blocks:
            blk.attention_pair_bias.force_bfloat16 = bf16
        ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if bf16
               else torch.autocast("cpu", enabled=False))
        with torch.no_grad(), ctx:
            out = ref(a_in.float(), s_in.float(), z_in.float(), None).float()
        for blk in ref.blocks:
            blk.attention_pair_bias.force_bfloat16 = True
        return out

    ceil = rel_rms(run(True).reshape(want.shape), run(False).reshape(want.shape))

    sd = {k[len(PREFIX):]: v.float()
          for k, v in torch.load(args.ckpt, map_location="cpu",
                                 weights_only=False)["model"].items()
          if k.startswith(PREFIX)}

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    def tt(x):
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)

    rows = []
    for arm in (True, False):
        dit = TokenDiffusionTransformer(sd, cfg, fp32_softmax=arm)
        got = torch.Tensor(ttnn.to_torch(
            dit(tt(a_in), tt(s_in), tt(z_in.unsqueeze(0))))).float().reshape(want.shape)
        e = rel_rms(got, want)
        rows.append({"fp32_softmax": arm, "pcc": round(pcc(got, want), 7),
                     "rel_rms": round(e, 6), "x_ceiling": round(e / ceil, 2)})
    print(json.dumps({"shape": list(want.shape), "bf16_ceiling": round(ceil, 6),
                      "arms": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

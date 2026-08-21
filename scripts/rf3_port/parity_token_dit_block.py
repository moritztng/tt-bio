#!/usr/bin/env python3
"""Score ONE token-DiT block against a per-block capture.

At 24 blocks a small per-block error and a large first-block error are
indistinguishable from the outside, so the stack-level score cannot say which one is
happening. This runs a single block.
"""
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
    ap.add_argument("--capture", default="/home/ttuser/rf3_ref_work/fi/rna_dit0.pt")
    ap.add_argument("--zero_z", action="store_true",
                    help="zero the pair track on BOTH sides: if the gap goes away the "
                         "precomputed-bias layout is the bug, if it stays it is not")
    ap.add_argument("--zero_s", action="store_true",
                    help="same for the single/conditioning track (AdaLN + output gates)")
    args = ap.parse_args()

    import ttnn
    from tt_bio.rf3.token_dit import TokenDiffusionTransformer
    from tt_bio.rf3.weights import load_reference
    from tt_bio.tenstorrent import get_device

    cap = torch.load(args.capture, weights_only=False)
    a_in, s_in, z_in = cap["in"][0], cap["in"][1], cap["in"][2]
    want = cap["out"]
    if args.zero_z:
        z_in = torch.zeros_like(z_in)
    if args.zero_s:
        s_in = torch.zeros_like(s_in)
    if args.zero_z or args.zero_s:
        want = None   # recomputed from the reference below

    net, _ = load_reference(args.ckpt, num_steps=2)
    blk = net.diffusion_module.diffusion_transformer.blocks[0]

    def ref(bf16):
        blk.attention_pair_bias.force_bfloat16 = bf16
        ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if bf16
               else torch.autocast("cpu", enabled=False))
        with torch.no_grad(), ctx:
            o = blk(a_in.float(), s_in.float(), z_in.float(), None).float()
        blk.attention_pair_bias.force_bfloat16 = True
        return o

    if want is None:
        want = ref(True)
    ceil = rel_rms(ref(True).reshape(want.shape), ref(False).reshape(want.shape))

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

    dit = TokenDiffusionTransformer(sd, cfg, n_block=1, fp32_softmax=True)
    got = torch.Tensor(ttnn.to_torch(
        dit(tt(a_in), tt(s_in), tt(z_in.unsqueeze(0))))).float().reshape(want.shape)
    e = rel_rms(got, want)
    print(json.dumps({"blocks": 1, "zero_z": args.zero_z, "zero_s": args.zero_s,
                      "shape": list(want.shape),
                      "pcc": round(pcc(got, want), 7),
                      "rel_rms": round(e, 6),
                      "bf16_ceiling": round(ceil, 6),
                      "x_ceiling": round(e / ceil, 2) if ceil else None}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

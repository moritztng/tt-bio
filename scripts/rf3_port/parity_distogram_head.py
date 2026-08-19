#!/usr/bin/env python3
"""Score RF3's distogram head, ceiling measured in the same run."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
PREFIX = "shadow.distogram_head."


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
    ap.add_argument("--capture", default="/home/ttuser/rf3_ref_work/fi/rna_disto.pt")
    args = ap.parse_args()

    import ttnn
    from tt_bio.rf3.distogram_head import DistogramHead
    from tt_bio.rf3.weights import load_reference
    from tt_bio.tenstorrent import get_device

    cap = torch.load(args.capture, weights_only=False)
    z = cap["in"][0]
    want = cap["out"]

    net, _ = load_reference(args.ckpt, num_steps=2)
    ref = net.distogram_head

    def run(bf16):
        ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if bf16
               else torch.autocast("cpu", enabled=False))
        with torch.no_grad(), ctx:
            return ref(z).float()

    ceil = rel_rms(run(True).reshape(want.shape), run(False).reshape(want.shape))

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    w = {k[len(PREFIX):]: v.float()
         for k, v in torch.load(args.ckpt, map_location="cpu",
                                weights_only=False)["model"].items()
         if k.startswith(PREFIX)}
    head = DistogramHead(w, cfg)
    zt = ttnn.from_torch(z.reshape(1, *z.shape[-3:]).float(), layout=ttnn.TILE_LAYOUT,
                         device=dev, dtype=ttnn.bfloat16)
    got = torch.Tensor(ttnn.to_torch(head(zt))).float().reshape(want.shape)

    # a wrong symmetrisation is the classic defect here, so score the alternatives too
    alts = {}
    with torch.no_grad(), torch.autocast("cpu", enabled=False):
        alts["mean_instead_of_sum"] = round(pcc(
            ref.predictor((z + z.transpose(-2, -3)) / 2).float(), want), 6)
        alts["no_symmetrize"] = round(pcc(ref.predictor(z).float(), want), 6)

    e = rel_rms(got, want)
    print(json.dumps({"tensor": "distogram_logits", "shape": list(want.shape),
                      "pcc": round(pcc(got, want), 7), "rel_rms": round(e, 6),
                      "ceiling": round(ceil, 6),
                      "x_ceiling": round(e / ceil, 2) if ceil else None,
                      "wrong_variants_for_contrast": alts}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Which softmax formulation conserves attention mass on RF3's logits?

`probe_apb_chain.py` shows the port's attention weights sum to 0.9757 per row, not 1, at
both 53 and 32 tokens (so it is not tile padding), while the reference sums to exactly 1.
That 2.4% deficit is a multiplicative deficit on every weight in the row, so it does not
cancel in `probs @ v`: it is the whole of the op's 13.43x and therefore the whole of the
s-track's 11x. This prices the formulations that could fix it, on the real logits.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:rf3-port-p3 \
        python3 scripts/rf3_port/probe_softmax_variants.py \
            --ckpt ~/rf3_ref_work/rf3_latest.ckpt \
            --golden ~/rf3_ref_work/trunk_io_ligands.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from scripts.rf3_port.probe_apb_stagewise import (  # noqa: E402
    HEAD_DIM, apb_inputs, ref_stages, rel_rms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--stack", default="shadow.recycler.pairformer_stack.0.")
    ap.add_argument("--crop", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    import ttnn
    from tt_bio.tenstorrent import get_device

    sd = {k[len(args.stack):]: v.float()
          for k, v in torch.load(args.ckpt, map_location="cpu",
                                 weights_only=False)["model"].items()
          if k.startswith(args.stack)}
    gold = torch.load(args.golden, weights_only=False)
    s, z = gold["in"]
    s = s.float().unsqueeze(0) if s.dim() == 2 else s.float()
    z = z.float().unsqueeze(0) if z.dim() == 3 else z.float()
    if args.crop:
        s, z = s[:, :args.crop], z[:, :args.crop, :args.crop]
    got, apb = apb_inputs(sd, s, z)
    hi = ref_stages(apb, got["in"][0].float(), got["in"][2].float(), bf16=False)

    # the softmax input the port builds, in the port's layout [B,H,I,J], fp32
    x = (hi["logits"] * HEAD_DIM ** -0.5 + hi["b"]).permute(0, 3, 1, 2).contiguous()
    ref = torch.softmax(x.double(), dim=-1).float()

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    def tt(t, dtype=ttnn.float32):
        return ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)

    def report(name, out):
        p = torch.Tensor(ttnn.to_torch(out)).float().reshape(ref.shape)
        rs = p.sum(dim=-1)
        return {"variant": name,
                "rowsum_mean": round(float(rs.mean()), 6),
                "rowsum_min": round(float(rs.min()), 6),
                "rowsum_max": round(float(rs.max()), 6),
                "rel_rms_vs_fp64_softmax": round(rel_rms(p, ref), 6)}

    rows = []
    rows.append(report("softmax_in_place (production)", ttnn.softmax_in_place(tt(x))))
    rows.append(report("softmax_in_place numeric_stable",
                       ttnn.softmax_in_place(tt(x), numeric_stable=True)))
    rows.append(report("softmax dim=-1", ttnn.softmax(tt(x), dim=-1)))
    rows.append(report("softmax dim=-1 numeric_stable",
                       ttnn.softmax(tt(x), dim=-1, numeric_stable=True)))
    rows.append(report("softmax dim=-1 + cfg",
                       ttnn.softmax(tt(x), dim=-1, compute_kernel_config=cfg)))
    rows.append(report("softmax dim=-1 numeric_stable + cfg",
                       ttnn.softmax(tt(x), dim=-1, numeric_stable=True,
                                    compute_kernel_config=cfg)))
    # bf16 storage, for the record: the reference's own softmax runs on bf16 logits
    rows.append(report("softmax bf16 in numeric_stable",
                       ttnn.softmax(tt(x, ttnn.bfloat16), dim=-1, numeric_stable=True)))

    def manual(t_in, use_cfg):
        kw = {"compute_kernel_config": cfg} if use_cfg else {}
        m = ttnn.max(t_in, dim=-1, keepdim=True)
        e = ttnn.exp(ttnn.subtract(t_in, m))
        sm = ttnn.sum(e, dim=-1, keepdim=True, **kw)
        return ttnn.divide(e, sm)

    rows.append(report("manual max/exp/sum/div fp32", manual(tt(x), False)))
    rows.append(report("manual max/exp/sum/div fp32 + cfg", manual(tt(x), True)))
    # renormalise the production softmax: one extra sum + divide over the same tensor
    pr = ttnn.softmax_in_place(tt(x))
    rows.append(report("production + renormalise",
                       ttnn.divide(pr, ttnn.sum(pr, dim=-1, keepdim=True,
                                                compute_kernel_config=cfg))))

    # Is this specific to RF3's logit range? Same op on a benign attention distribution
    # (within-row spread ~1, which is what every other model in the repo feeds it).
    gen = torch.Generator().manual_seed(0)
    for spread in (1.0, 10.0, 50.0, 135.0):
        y = torch.randn(x.shape, generator=gen) * spread
        yp = torch.Tensor(ttnn.to_torch(ttnn.softmax_in_place(tt(y)))).float()
        rs = yp.sum(dim=-1)
        rows.append({"variant": f"production, synthetic within-row std {spread:g}",
                     "rowsum_mean": round(float(rs.mean()), 6),
                     "rowsum_min": round(float(rs.min()), 6),
                     "rowsum_max": round(float(rs.max()), 6),
                     "rel_rms_vs_fp64_softmax": round(
                         rel_rms(yp, torch.softmax(y.double(), dim=-1).float()), 6)})

    rep = {"tokens": int(x.shape[-1]), "crop": args.crop,
           "input_absmax": round(float(x.abs().max()), 2),
           "input_within_row_std": round(float((x - x.mean(-1, keepdim=True)).std()), 2),
           "variants": rows}
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

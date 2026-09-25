#!/usr/bin/env python3
"""What makes `ttnn.layer_norm` read 3-4x the bf16 floor on the msa_module pair states.

Same weights, config and bf16 input as pair_transition's LayerNorm, against the float64 LN of the
float64 state. Variants: the shipped call; Welford; the non-legacy flags spelled out; the input
centred on the host before upload (removes the row mean, so a mean-dominated cancellation shows);
an fp32 input (whose output is fp32 too).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inter", type=Path, required=True)
    ap.add_argument("--boundary", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(os.path.expanduser("~/of3-weights/of3-p2-155k.pt")))
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    import ttnn
    from tt_bio import tenstorrent as TT
    from tt_bio.openfold3_msa_embedder import MSAModule
    from tt_bio.openfold3_weights import is_openbind

    I = torch.load(a.inter, map_location="cpu", weights_only=False)
    B = torch.load(a.boundary, map_location="cpu", weights_only=False)
    n = int(B["outputs"].shape[-2])
    tokm = torch.diagonal(B["inputs"]["kwargs"]["pair_mask"].reshape(n, n)) > 0
    rz = lambda x: x.reshape(n, n, -1)[tokm][:, tokm]
    rel = lambda x, r: float((rz(x) - rz(r)).norm() / rz(r).norm())
    sd = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    dev = TT.get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    msa = MSAModule(sd, ckc, transpose_bias=not is_openbind(sd))
    del sd
    up = lambda x, dt=ttnn.bfloat16: ttnn.from_torch(x.float().reshape(1, n, n, -1),
                                                     layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
    dn = lambda t: torch.Tensor(ttnn.to_torch(t)).double().reshape(1, n, n, -1)
    rep = {"host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"), "blocks": {}}
    for i, blk in enumerate(msa.blocks):
        tr = blk.pair_stack.transition_z
        x, ref = I[i]["x"], I[i]["ln"]
        g = torch.Tensor(ttnn.to_torch(tr.norm_weight)).double().reshape(-1)[:x.shape[-1]]
        be = torch.Tensor(ttnn.to_torch(tr.norm_bias)).double().reshape(-1)[:x.shape[-1]]
        mu = x.mean(-1, keepdim=True)
        sdv = x.var(-1, unbiased=False, keepdim=True).sqrt()
        # float64 LN with the DEVICE's bf16 weights: separates weight rounding from the kernel.
        ref_bw = (x - mu) / (sdv ** 2 + 1e-5).sqrt() * g + be
        L = lambda xin, **k: dn(ttnn.layer_norm(xin, weight=tr.norm_weight, bias=tr.norm_bias,
                                                epsilon=1e-5, compute_kernel_config=ckc, **k))
        PC = ttnn.LayerNormDefaultProgramConfig
        g_ = dev.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                                ttnn.CoreCoord(g_.x - 1, g_.y - 1))})
        rec_ = ttnn.create_layer_norm_reciprocals(dev, crs, int(x.shape[-1]))
        v = {"shipped": L(up(x)),
             "welford": L(up(x), program_config=PC(use_welford=True), recip_tensor=rec_),
             "legacy_reduction+rsqrt": L(up(x), program_config=PC(legacy_reduction=True,
                                                                   legacy_rsqrt=True)),
             "host_centred_input": L(up(x - mu)),
             "fp32_input": L(up(x, ttnn.float32)),
             "fp32_input_welford": L(up(x, ttnn.float32), program_config=PC(use_welford=True),
                                       recip_tensor=rec_)}
        # gamma/beta at fp32 from the checkpoint, laid out as the shipped bf16 handles are.
        wl = lambda t, like: ttnn.from_torch(t.float().reshape(tuple(like.shape)), dtype=ttnn.float32,
                                             layout=like.layout, device=dev)
        g32, b32 = wl(I[i]["ln_w"], tr.norm_weight), wl(I[i]["ln_b"], tr.norm_bias)
        L32 = lambda xin: dn(ttnn.layer_norm(xin, weight=g32, bias=b32, epsilon=1e-5,
                                             compute_kernel_config=ckc))
        v["fp32_weights"] = L32(up(x))
        v["fp32_weights_fp32_input"] = L32(up(x, ttnn.float32))
        row = {k: rel(t, ref) for k, t in v.items()}
        row["bf16_floor"] = rel(ref.to(torch.bfloat16).double(), ref)
        row["f64_ln_with_device_bf16_weights_vs_ref"] = rel(ref_bw, ref)
        row["mean_over_std_rms"] = float((mu / sdv).pow(2).mean().sqrt())
        row["bf16_input_rounding_alone"] = rel(
            ((b := x.to(torch.bfloat16).double()) - b.mean(-1, keepdim=True))
            / (b.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt() * g + be, ref)
        rep["blocks"][i] = row
        print(f"b{i}: " + "  ".join(f"{k} {r:.3e}" for k, r in row.items()), flush=True)
    a.report.write_text(json.dumps(rep, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

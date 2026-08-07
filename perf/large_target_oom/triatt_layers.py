#!/usr/bin/env python3
"""Layer probe: per-chunk layer_norm and the triangle bias, chunked vs full reference.

Runs inside a tt-bio checkout. Uses the same TriangleAttention weights as
triatt_probe (opendde refiner block 0) but drives the class internals directly.
"""
import torch

import ttnn
from tt_bio import tenstorrent as T

CKPT = ("/home/ttuser/.cache/huggingface/hub/models--aurekaresearch--OpenDDE/"
        "snapshots/eddd563ce96571f784012edd8f045181c8f8627d/opendde_abag.pt")
N = 1712


def cmp(name, a, b):
    d = (a.float() - b.float()).abs()
    pcc = torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()
    print(f"{name}: {'BIT-EXACT' if torch.equal(a, b) else 'DIFFERS'} "
          f"maxabs {d.max().item():.4e} PCC {pcc:.8f}", flush=True)


def main():
    from tt_bio.opendde import load_opendde_checkpoint, route_opendde_weights
    routed = route_opendde_weights(load_opendde_checkpoint(CKPT, abag=True))
    blk = {k[len("layers.0."):]: v for k, v in routed["refiner"].items()
           if k.startswith("layers.0.")}

    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    torch.manual_seed(0)
    z_host = torch.randn(1, N, N, 384)

    for ending in (False, True):
        scope = "tri_att_end" if ending else "tri_att_start"
        w = T.WeightScope(blk).child(scope, "mha.")
        ta = T.TriangleAttention(384 // 12, 12, ending, w, ckc)
        z = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        x = ttnn.reshape(z, (N, N, 384))
        chunk = 512

        # reference: full permute + ln
        xf = ttnn.permute(x, (1, 0, 2)) if ending else x
        ln_full = ttnn.layer_norm(xf, weight=ta.layer_norm_weight, bias=ta.layer_norm_bias,
                                  epsilon=1e-5, compute_kernel_config=ckc)
        # per-chunk, my way
        for s in (0, 512, 1024):
            e = min(s + chunk, N)
            blk_t = x[:, s:e, :] if ending else x[s:e, :, :]
            if ending:
                blk_t = ttnn.permute(blk_t, (1, 0, 2))
            ln_c = ttnn.layer_norm(blk_t, weight=ta.layer_norm_weight, bias=ta.layer_norm_bias,
                                   epsilon=1e-5, compute_kernel_config=ckc)
            cmp(f"ending={ending} ln chunk {s}",
                ttnn.to_torch(ln_full[s:e]), ttnn.to_torch(ln_c))
            # qkv per chunk vs sliced full
            qkv_full = ttnn.experimental.minimal_matmul(ln_full, ta.qkv_weight,
                                                        compute_kernel_config=ckc, dtype=T._dtype())
            qkv_c = ttnn.experimental.minimal_matmul(ln_c, ta.qkv_weight,
                                                     compute_kernel_config=ckc, dtype=T._dtype())
            cmp(f"ending={ending} qkv chunk {s}",
                ttnn.to_torch(qkv_full[s:e]), ttnn.to_torch(qkv_c))
            ttnn.deallocate(qkv_full)
            ttnn.deallocate(qkv_c)
            ttnn.deallocate(ln_c)
        # bias: full vs per-chunk assembled
        b_full = ttnn.linear(ln_full, ta.bias_weight, compute_kernel_config=ckc,
                             dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN)
        b_full = ttnn.permute(ttnn.unsqueeze(b_full, 0), (0, 3, 1, 2))
        parts = []
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            blk_t = x[:, s:e, :] if ending else x[s:e, :, :]
            if ending:
                blk_t = ttnn.permute(blk_t, (1, 0, 2))
            xc = ttnn.layer_norm(blk_t, weight=ta.layer_norm_weight, bias=ta.layer_norm_bias,
                                 epsilon=1e-5, compute_kernel_config=ckc)
            ttnn.deallocate(blk_t) if ending else None
            b = ttnn.linear(xc, ta.bias_weight, compute_kernel_config=ckc,
                            dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN)
            ttnn.deallocate(xc)
            bp = ttnn.unsqueeze(b, 0)
            ttnn.deallocate(b)
            parts.append(ttnn.permute(bp, (0, 3, 1, 2)))
            ttnn.deallocate(bp)
        b_asm = ttnn.concat(parts, dim=2)
        cmp(f"ending={ending} bias", ttnn.to_torch(b_full), ttnn.to_torch(b_asm))
        ttnn.deallocate(z)
        ttnn.deallocate(x) if not ending else None


if __name__ == "__main__":
    main()

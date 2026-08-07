#!/usr/bin/env python3
"""Replicate main's chunked tri_att path and the F3 path side by side, tap by tap.

Usage: TT_VISIBLE_DEVICES=3 python3 triatt_tap.py   (inside a tt-bio checkout)
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
    chunk = 512

    for ending in (False, True):
        scope = "tri_att_end" if ending else "tri_att_start"
        ta = T.TriangleAttention(384 // 12, 12, ending, T.WeightScope(blk).child(scope, "mha."), ckc)

        def attend(qkv_in, bias):
            qkv_in = ttnn.unsqueeze(qkv_in, 1)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv_in, num_heads=ta.n_heads, num_kv_heads=ta.n_heads,
                transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(qkv_in)
            o = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, is_causal=False, scale=ta.scale ** -1,
                program_config=T._tri_att_sdpa_program_config(q.shape[2], k.shape[2]))
            ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
            o_heads = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(o)
            return ttnn.squeeze(o_heads, 1)

        def gap(o_in, g_in):
            o_in = ttnn.multiply_(o_in, g_in, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            ttnn.deallocate(g_in)
            x_out = ttnn.linear(o_in, ta.o_weight, compute_kernel_config=ckc,
                                dtype=T._dtype(), core_grid=T.CORE_GRID_MAIN)
            ttnn.deallocate(o_in)
            return x_out

        z1 = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        z2 = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

        # ---- path A: main's chunked path (full ln, full bias, slice per chunk)
        xa = ttnn.reshape(z1, (N, N, 384))
        if ending:
            xa = ttnn.permute(xa, (1, 0, 2))
        xa = ttnn.layer_norm(xa, weight=ta.layer_norm_weight, bias=ta.layer_norm_bias,
                             epsilon=1e-5, compute_kernel_config=ckc)
        bias_a = ttnn.linear(xa, ta.bias_weight, compute_kernel_config=ckc,
                             dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN)
        bias_a = ttnn.permute(ttnn.unsqueeze(bias_a, 0), (0, 3, 1, 2))
        parts_a = []
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            xc = xa[s:e, :, :]
            qkv = ttnn.experimental.minimal_matmul(xc, ta.qkv_weight, compute_kernel_config=ckc, dtype=T._dtype())
            g = ttnn.experimental.minimal_matmul(xc, ta.g_weight, compute_kernel_config=ckc, dtype=T._dtype())
            o = attend(qkv, bias_a)
            ttnn.deallocate(qkv)
            parts_a.append(gap(o, g))
        # ---- path B: F3 (per-chunk recompute)
        xb = ttnn.reshape(z2, (N, N, 384))

        def normed_rows(s, e):
            b = xb[:, s:e, :] if ending else xb[s:e, :, :]
            if ending:
                b = ttnn.permute(b, (1, 0, 2))
            return ttnn.layer_norm(b, weight=ta.layer_norm_weight, bias=ta.layer_norm_bias,
                                   epsilon=1e-5, compute_kernel_config=ckc)
        bp_ = []
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            xc = normed_rows(s, e)
            b = ttnn.linear(xc, ta.bias_weight, compute_kernel_config=ckc,
                            dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN)
            ttnn.deallocate(xc)
            t = ttnn.unsqueeze(b, 0)
            ttnn.deallocate(b)
            bp_.append(ttnn.permute(t, (0, 3, 1, 2)))
            ttnn.deallocate(t)
        bias_b = ttnn.concat(bp_, dim=2)
        cmp(f"ending={ending} triangle_bias", ttnn.to_torch(bias_a), ttnn.to_torch(bias_b))
        parts_b = []
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            xc = normed_rows(s, e)
            qkv = ttnn.experimental.minimal_matmul(xc, ta.qkv_weight, compute_kernel_config=ckc, dtype=T._dtype())
            g = ttnn.experimental.minimal_matmul(xc, ta.g_weight, compute_kernel_config=ckc, dtype=T._dtype())
            ttnn.deallocate(xc)
            o = attend(qkv, bias_b)
            ttnn.deallocate(qkv)
            parts_b.append(gap(o, g))
        for i, (pa, pb) in enumerate(zip(parts_a, parts_b)):
            cmp(f"ending={ending} part{i}", ttnn.to_torch(pa), ttnn.to_torch(pb))


if __name__ == "__main__":
    main()

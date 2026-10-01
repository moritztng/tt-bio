#!/usr/bin/env python3
"""C9's two backward paths graded against float64 at the op.

`triatt_qkv_heads` has two VJPs: the packed sink (triatt_bw's QKV_PACKED write, the path the block
A/B reached 16 of 16 times) and the per-slot path (a q, k or v slot that receives its own
gradient, which the block never reaches). Both are graded here on the same inputs against a torch
float64 reference of Y = X @ W split into heads: dX = G @ W^T, dW = X^T G with G the packed
cotangent. The composed path (entry off: minimal_matmul + nlp_create_qkv_heads) is graded beside
them as the bar. Run with TT_BIO_TAPED_KERNELS naming triatt_qkv_heads."""
import json, os, pathlib, sys
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a - b).norm() / b.norm())


def main():
    out = sys.argv[1]
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio.tenstorrent import get_device, _qkv_mm_config
    from tt_bio import autograd as ag, taped_ttnn as TT, triatt_qkv as Q
    dev = get_device()
    torch.manual_seed(0)
    B, L, C, H, dh = 288, 288, 128, 8, 32
    W = 3 * H * dh
    x64 = torch.randn(B, L, C, dtype=torch.float64)
    w64 = torch.randn(C, W, dtype=torch.float64) * C ** -0.5
    g64 = [torch.randn(B, H, L, dh, dtype=torch.float64) * 1e-2 for _ in range(3)]
    up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=dev)
    # The reference uses the bf16-rounded operands the device sees, so the grade is the op's error.
    xb, wb = x64.to(torch.bfloat16).double(), w64.to(torch.bfloat16).double()
    gb = [g.to(torch.bfloat16).double() for g in g64]
    G = torch.cat([g.permute(0, 2, 1, 3).reshape(B, L, H * dh) for g in gb], -1)
    ref_dx, ref_dw = G @ wb.T, xb.reshape(-1, C).T @ G.reshape(-1, W)
    ckc = ag.precise_config()
    res = {"shape": [B, L, C, H, dh]}

    def run(arm):
        os.environ["TT_BIO_TAPED_KERNELS"] = (
            TT.TAPED_KERNELS_DEFAULT + ",triatt_qkv_heads" if arm != "composed"
            else TT.TAPED_KERNELS_DEFAULT).strip(",")
        TT.KERNEL_STATS.pop("triatt_qkv_heads", None)
        xv, wv = up(x64), up(w64)
        xt, wt = ag.Tensor(xv, requires_grad=True), ag.Tensor(wv, requires_grad=True)
        with TT.tape():
            cfg = _qkv_mm_config(xv, wv)
            qkv = Q.qkv_heads(xt, wt, ckc, H, dh, ttnn.bfloat16, cfg)
            if qkv is None:
                y = ttnn.experimental.minimal_matmul(input_tensor=xt, weight_tensor=wt,
                                                     compute_kernel_config=ckc,
                                                     dtype=ttnn.bfloat16, config=cfg)
                qkv = ttnn.experimental.nlp_create_qkv_heads(
                    ttnn.unsqueeze(y, 1), num_heads=H, num_kv_heads=H,
                    transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            q, k, v = (ag._wrap(t) for t in qkv)
            gd = [up(g) for g in g64]
            if arm == "packed":
                sink = ag._packed_qkv_source(q, k, v)
                assert sink is not None and not isinstance(sink, ag.Tensor), sink
                sink.add_grad(ttnn.concat([ttnn.reshape(ag.merge_heads_value(g), [B, 1, L, H * dh])
                                           for g in gd], dim=-1))
            else:
                ag.backward([q, k, v], gd)
        dx = ttnn.to_torch(xt.grad).double()
        dw = ttnn.to_torch(wt.grad).double()
        return {"entry_served_declined": TT.KERNEL_STATS.get("triatt_qkv_heads", [0, 0]),
                "dx_rel_l2": rel(dx, ref_dx), "dw_rel_l2": rel(dw, ref_dw),
                "dx_max_abs": float((dx - ref_dx).abs().max()),
                "dw_max_abs": float((dw - ref_dw).abs().max())}

    for arm in ("composed", "per_slot", "packed"):
        res[arm] = run(arm)
        print(arm, json.dumps(res[arm]), flush=True)
    pathlib.Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

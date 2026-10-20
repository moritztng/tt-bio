"""mm_rope's half-pair arm for heads narrower than 32 (SaProt 35M: 20 heads of 24): qkv + bias + NeoX RoPE + head split
in one minimal_matmul (MM_HALF_PAIRS writer), q / k / v [B, H, L, 32] against float64 on the same bf16 inputs, and its
time beside the plain qkv minimal_matmul it replaces (the split, reshapes and RL it also replaces are in the page census).
Batch 1 keeps M (544) under N (1920): plan() declines M > N until the transposed grid is handled.

    TT_VISIBLE_DEVICES=N python perf/spd_m_saprot/half_pair_check.py --out check.jsonl [--fast] [--batch 1]
"""
import argparse
import json
import math
import time

import torch
import ttnn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grid", default="5,8")
    ap.add_argument("--cfg", default="3,5,4,1,4")
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    from tt_bio import tenstorrent as T, mm_rope as MR
    T.set_fast_mode(a.fast)
    dev = T.get_device()
    D, H, hd, B, L = 480, 20, 24, a.batch, 544
    cfg = tuple(int(v) for v in a.cfg.split(","))
    grid = tuple(int(v) for v in a.grid.split(","))
    gen = torch.Generator().manual_seed(0)
    xt = torch.randn(B, L, D, generator=gen).to(torch.bfloat16)
    w = torch.randn(3 * H * hd, D, generator=gen, dtype=torch.float64) / math.sqrt(D)
    bias = torch.randn(3 * H * hd, generator=gen, dtype=torch.float64) * 0.1
    wh, bh = MR.half_pairs(w, bias, H, hd)
    wdt = ttnn.bfloat8_b if a.fast else ttnn.bfloat16
    tt = lambda t, dt: ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
    x = tt(xt, ttnn.bfloat16)
    wp_t, bp_t = MR.permute(wh, grid[0], cfg[2]), MR.permute(bh, grid[0], cfg[2])
    wp, bp = tt(wp_t, wdt), tt(bp_t, ttnn.bfloat16)
    wplain, bplain = tt(wh, wdt), tt(bh, ttnn.bfloat16)
    cos_t, sin_t = MR.half_pair_tables(L, hd)
    cos, sin = tt(cos_t[None, None], ttnn.bfloat16), tt(sin_t[None, None], ttnn.bfloat16)
    fid = ttnn.MathFidelity.HiFi3 if a.fast else ttnn.MathFidelity.HiFi4
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True,
                                          packer_l1_acc=True)
    mcfg = ttnn.MinimalMatmulConfig(M_block_size=cfg[0], K_block_size=cfg[1], N_block_size=cfg[2], subblock_h=cfg[3],
                                    subblock_w=cfg[4], compute_with_storage_grid_size=ttnn.CoreCoord(*grid))

    # float64 on the device's inputs: bf16 x, the weight as stored (bf16 or block-fp8), bf16 bias and tables
    w64 = ttnn.to_torch(wplain).double()
    ref = MR.half_pair_heads(xt.double() @ w64 + bh.to(torch.bfloat16).double(), cos_t.to(torch.bfloat16).double(),
                             sin_t.to(torch.bfloat16).double(), H)
    rope = lambda: MR.qkv_rope_heads(x, wp, bp, cos, sin, H, cfg, grid, ck, half=True)
    plain = lambda: ttnn.experimental.minimal_matmul(x, wplain, bias_tensor=bplain, compute_kernel_config=ck, config=mcfg,
                                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rec = dict(fast=a.fast, batch=B, grid=a.grid, cfg=a.cfg, plan=MR.plan(x, wp, cfg, grid, L))
    o = rope()
    if o is not None:
        o = [ttnn.to_torch(t).double() for t in o]
        rec.update(shapes=[list(t.shape) for t in o],
                   rel_l2=[float((u - r).norm() / r.norm()) for u, r in zip(o, ref)],
                   pad_max=[float(t[..., [*range(12, 16), *range(28, 32)]].abs().max()) for t in o[:2]],
                   qk_rel_l2=float(((o[0] @ o[1].transpose(-1, -2)) - ref[0] @ ref[1].transpose(-1, -2)).norm()
                                   / (ref[0] @ ref[1].transpose(-1, -2)).norm()))
    for name, fn in (("rope", rope), ("plain", plain)) if o is not None else (("plain", plain),):
        for _ in range(2):
            r = fn()
            for t in (r if isinstance(r, (list, tuple)) else [r]):
                ttnn.deallocate(t)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(a.calls):
            r = fn()
            for t in (r if isinstance(r, (list, tuple)) else [r]):
                ttnn.deallocate(t)
        ttnn.synchronize_device(dev)
        rec[f"{name}_ms"] = round((time.perf_counter() - t0) / a.calls * 1e3, 4)
    print(json.dumps(rec), flush=True)
    open(a.out, "a").write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

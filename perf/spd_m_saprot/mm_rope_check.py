"""SaProt's qkv projection written straight into per-head q, k, v ([B, H, L, 64]) by the minimal_matmul writer
(kernels/mm_gate, MM_HEADS_ST), against minimal_matmul + nlp_create_qkv_heads. The page shape, the shipped WH qkv
block config, with the qkv bias; the output should be bit-identical and the split op's time gone.

    TT_VISIBLE_DEVICES=N python perf/spd_m_saprot/mm_rope_check.py --model saprot-650m [--fast] --out check.jsonl
"""
import argparse
import json
import math
import time

import torch
import ttnn

MODELS = {"saprot-650m": (1280, 20), "saprot-1.3b": (1280, 20)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saprot-650m", choices=sorted(MODELS))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--cfg", default="8,8,8,1,4")
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    from tt_bio import tenstorrent as T, mm_generic as MG, mm_gate
    T.set_fast_mode(a.fast)
    dev = T.get_device()
    d, H = MODELS[a.model]
    B, L = 7, 544
    cfg = tuple(int(v) for v in a.cfg.split(","))
    grid = tuple(T.COMPUTE_GRID_MAIN)
    gen = torch.Generator().manual_seed(0)
    x = ttnn.from_torch(torch.randn(B, L, d, generator=gen).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    w = ttnn.from_torch((torch.randn(d, 3 * d, generator=gen) / math.sqrt(d)).to(torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat8_b if a.fast else ttnn.bfloat16)
    b = ttnn.from_torch((torch.randn(1, 3 * d, generator=gen) * 0.1).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16)
    fid = ttnn.MathFidelity.HiFi3 if a.fast else ttnn.MathFidelity.HiFi4
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True,
                                          packer_l1_acc=True)
    mcfg = ttnn.MinimalMatmulConfig(M_block_size=cfg[0], K_block_size=cfg[1], N_block_size=cfg[2], subblock_h=cfg[3],
                                    subblock_w=cfg[4], compute_with_storage_grid_size=ttnn.CoreCoord(*grid))

    def ref():
        qkv = ttnn.experimental.minimal_matmul(x, w, bias_tensor=b, compute_kernel_config=ck, config=mcfg,
                                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(ttnn.unsqueeze(qkv, 1), num_heads=H, num_kv_heads=H,
                                                          transpose_k_heads=False,
                                                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(qkv)
        return q, k, v

    def heads():
        outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([B, H, L, 64]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                               ttnn.DRAM_MEMORY_CONFIG) for _ in range(3)]
        return MG.generic_minimal_matmul(dev, x, w, outs, (cfg, grid), MG.ckc_args(ck),
                                         defines={"MM_HEADS_ST": L // 32}, kernel_dir=mm_gate.KERNEL_DIR, bias=b)

    def plain():
        return MG.generic_minimal_matmul(dev, x, w, [ttnn.allocate_tensor_on_device(
            ttnn.Shape([B, L, 3 * d]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)],
            (cfg, grid), MG.ckc_args(ck), kernel_dir=mm_gate.KERNEL_DIR, bias=b)

    out = open(a.out, "a")
    r = [ttnn.to_torch(t) for t in ref()]
    h = [ttnn.to_torch(t) for t in heads()]
    p = ttnn.to_torch(plain())
    rq = torch.cat([t.permute(0, 2, 1, 3).reshape(B, L, d) for t in r], -1)
    rec = dict(model=a.model, fast=a.fast, cfg=a.cfg, heads_equal=[bool(torch.equal(u, v)) for u, v in zip(r, h)],
               heads_maxdiff=[float((u.float() - v.float()).abs().max()) for u, v in zip(r, h)],
               plain_vs_ref_equal=bool(torch.equal(p, rq)))
    for name, fn in (("ref", ref), ("heads", heads), ("plain", plain)):
        for _ in range(2):
            o = fn()
            for t in (o if isinstance(o, (list, tuple)) else [o]):
                ttnn.deallocate(t)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(a.calls):
            o = fn()
            for t in (o if isinstance(o, (list, tuple)) else [o]):
                ttnn.deallocate(t)
        ttnn.synchronize_device(dev)
        rec[f"{name}_ms"] = round((time.perf_counter() - t0) / a.calls * 1e3, 4)
    print(json.dumps(rec), flush=True)
    out.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

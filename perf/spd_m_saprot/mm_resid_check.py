"""SaProt's out projection and fc2 with the block's residual add in the matmul's epilogue (tt_bio/mm_resid), against
today's path: the picked minimal_matmul with bias, then ttnn.add with the residual (into L1 under --fast, as
_SAPROT_X_L1 does). Page shape (7 x 544 rows), the model's kernel config per projection, both against float64.

    TT_VISIBLE_DEVICES=N python perf/spd_m_saprot/mm_resid_check.py --model saprot-650m [--fast] --out check.jsonl
"""
import argparse
import json
import math
import time

import torch
import ttnn

MODELS = {"saprot-35m": (480, 1920), "saprot-650m": (1280, 5120), "saprot-1.3b": (1280, 5120)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saprot-650m", choices=sorted(MODELS))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    from tt_bio import tenstorrent as T, mm_resid as MR
    from tt_bio import saprot as S
    T.set_fast_mode(a.fast)
    dev = T.get_device()
    cfg, _ = S.CONFIGS[a.model]
    ck = S.Saprot(**cfg).compute_kernel_config
    d, ff = MODELS[a.model]
    dh = d if (d // cfg["n_heads"]) % 64 == 0 else cfg["n_heads"] * 32
    wdt = ttnn.bfloat8_b if a.fast else ttnn.bfloat16
    rmc = ttnn.L1_MEMORY_CONFIG if a.fast else ttnn.DRAM_MEMORY_CONFIG
    B, L = 7, 544
    gen = torch.Generator().manual_seed(0)
    out = open(a.out, "a")
    for name, K in (("out", dh), ("fc2", ff)):
        x = torch.randn(B, L, K, generator=gen).to(torch.bfloat16)
        w = (torch.randn(K, d, generator=gen) / math.sqrt(K)).to(torch.bfloat16)
        b = (torch.randn(1, d, generator=gen) * 0.1).to(torch.bfloat16)
        r = (torch.randn(B, L, d, generator=gen) * 2).to(torch.bfloat16)
        xd = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        wd = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=wdt)
        bd = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        rd = ttnn.from_torch(r, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=rmc)
        lin = S._SaprotLin.__new__(S._SaprotLin)
        lin.compute_kernel_config, lin.norm_weight = ck, torch.empty(d)
        mck = lin._ck(wd)
        prog = S._saprot_prog(xd, wd, None, mck.math_fidelity == ttnn.MathFidelity.HiFi2)
        rec = dict(model=a.model, fast=a.fast, shape=name, prog=None if prog is None else prog[0],
                   fidelity=str(mck.math_fidelity))
        if prog is None or prog[0] != "mm":
            print(json.dumps(rec), flush=True)
            out.write(json.dumps(rec) + "\n")
            continue
        c = prog[1]
        g = c.compute_with_storage_grid_size
        bc = (c.M_block_size, c.K_block_size, c.N_block_size, c.subblock_h, c.subblock_w)
        rec["cfg"] = bc

        def ref():
            p = ttnn.experimental.minimal_matmul(xd, wd, bias_tensor=bd, compute_kernel_config=mck, dtype=ttnn.bfloat16,
                                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, config=c)
            o = ttnn.add(rd, p, memory_config=rmc)
            ttnn.deallocate(p)
            return o

        def fused():
            return MR.matmul_add(xd, wd, bd, rd, bc, (g.x, g.y), mck, memory_config=rmc)

        wq = ttnn.to_torch(wd).double()   # the weight as the device holds it (block-fp8 under --fast)
        exact = r.double() + x.double() @ wq + b.double()
        sc = exact.norm()
        for arm, fn in (("ref", ref), ("fused", fused)):
            o = ttnn.to_torch(fn()).double()
            e = o - exact
            rec[f"{arm}_rel_l2"] = float(e.norm() / sc)
            rec[f"{arm}_bias"] = float(e.mean() / exact.abs().mean())
            rec[f"{arm}_max_abs"] = float(e.abs().max())
            rec[f"_{arm}"] = o
        rec["ndiff"] = int((rec.pop("_ref") != rec.pop("_fused")).sum())
        for arm, fn in (("ref", ref), ("fused", fused)):
            for _ in range(2):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(a.calls):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(dev)
            rec[f"{arm}_ms"] = round((time.perf_counter() - t0) / a.calls * 1e3, 4)
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n")
        for t in (xd, wd, bd, rd):
            ttnn.deallocate(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""SaProt's fc1 with its fused GELU: today's 2D program against minimal_matmul at the given blocks (_SAPROT_FC1_MM),
page shape (7 x 544 rows), both against float64 gelu(x @ w + b) with the weight as the device holds it.

    TT_VISIBLE_DEVICES=N python perf/spd_m_saprot/fc1_check.py --model saprot-650m [--fast] --out check.jsonl
"""
import argparse
import json
import math
import time

import torch
import ttnn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saprot-650m")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--cfgs", default="8,8,8,1,4;4,8,8,1,4;4,4,8,1,4;8,4,8,1,4;8,2,8,1,4")
    ap.add_argument("--where", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    from tt_bio import tenstorrent as T
    from tt_bio import saprot as S
    T.set_fast_mode(a.fast)
    dev = T.get_device()
    cfg, _ = S.CONFIGS[a.model]
    ck = S.Saprot(**cfg).compute_kernel_config
    d = {"saprot-35m": 480}.get(a.model, 1280)
    ff = 4 * d
    act = S._fc1_act()
    wdt = ttnn.bfloat8_b if a.fast else ttnn.bfloat16
    B, L = 7, 544
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(B, L, d, generator=gen).to(torch.bfloat16)
    w = (torch.randn(d, ff, generator=gen) / math.sqrt(d) * 2).to(torch.bfloat16)
    b = (torch.randn(1, ff, generator=gen) * 0.1).to(torch.bfloat16)
    xd = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    wd = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=wdt)
    bd = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    lin = S._SaprotLin.__new__(S._SaprotLin)
    lin.compute_kernel_config, lin.norm_weight = ck, torch.empty(d)
    mck = lin._ck(wd)
    hifi2 = mck.math_fidelity == ttnn.MathFidelity.HiFi2
    S._SAPROT_FC1_MM = False
    prog = S._saprot_prog(xd, wd, act, hifi2)
    rec = dict(model=a.model, fast=a.fast, act=act, fidelity=str(mck.math_fidelity), prog=None if prog is None else prog[0])

    def two_d():
        return ttnn.linear(xd, wd, bias=bd, program_config=prog[1], compute_kernel_config=mck, dtype=ttnn.bfloat16,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)

    fns = {"2d": two_d} if prog is not None and prog[0] == "2d" else {}
    for c in filter(None, a.cfgs.split(";")):
        bc = tuple(int(v) for v in c.split(","))
        mc = ttnn.MinimalMatmulConfig(M_block_size=bc[0], K_block_size=bc[1], N_block_size=bc[2], subblock_h=bc[3],
                                      subblock_w=bc[4], compute_with_storage_grid_size=dev.compute_with_storage_grid_size())
        fns["mm" + "_".join(map(str, bc))] = lambda mc=mc: ttnn.experimental.minimal_matmul(
            xd, wd, bias_tensor=bd, compute_kernel_config=mck, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            config=mc, fused_activation=(ttnn.UnaryOpType.GELU, act == "gelu_approx"))

    pre = x.double() @ ttnn.to_torch(wd).double() + b.double()
    exact = torch.nn.functional.gelu(pre)
    sc = exact.norm()
    for arm, fn in list(fns.items()):
        try:
            o = ttnn.to_torch(fn()).double()
        except RuntimeError as exc:
            rec[f"{arm}_error"] = str(exc).splitlines()[0][:200]
            del fns[arm]
            continue
        e = o - exact
        ae = e.abs()
        rec[f"{arm}_rel_l2"] = float(e.norm() / sc)
        rec[f"{arm}_bias"] = float(e.mean() / exact.abs().mean())
        rec[f"{arm}_max_abs"] = float(ae.max())
        rec[f"{arm}_n_gt_0p25"] = int((ae > 0.25).sum())
        for flat in ae.flatten().topk(a.where).indices.tolist():
            i = tuple(int(v) for v in torch.unravel_index(torch.tensor(flat), ae.shape))
            rec.setdefault(f"{arm}_worst", []).append([list(i), float(o[i]), float(exact[i]), float(pre[i])])
    for arm, fn in fns.items():
        for _ in range(2):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(a.calls):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        rec[f"{arm}_ms"] = round((time.perf_counter() - t0) / a.calls * 1e3, 4)
    print(json.dumps(rec), flush=True)
    open(a.out, "a").write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

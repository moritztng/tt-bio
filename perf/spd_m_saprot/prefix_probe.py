"""The K prefix at which fc1's normal-mode wrong element goes wrong (rowcol_probe: [2,447,2411] off by exactly -4.0 at
HiFi4 + fp32 dest + packer_l1_acc, right with pl1 off). pl1_probe's identity sums never wrap, so the fault needs real
HiFi4 products: here x_tile[:, :32k] @ w_tile[:32k] for k = 1..40 on ttnn.matmul, per fidelity and pl1, with the
element's device value against float64, so the first k where the -2^n offset appears names the add.

    TT_VISIBLE_DEVICES=N python perf/spd_m_saprot/prefix_probe.py --out probe.jsonl
"""
import argparse
import json
import math

import torch
import ttnn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    from tt_bio import tenstorrent as T
    T.set_fast_mode(False)
    dev = T.get_device()
    d, ff, B, L = 1280, 5120, 7, 544
    gen = torch.Generator().manual_seed(0)   # fc1_check's x and w
    x = torch.randn(B, L, d, generator=gen).to(torch.bfloat16)
    w = (torch.randn(d, ff, generator=gen) / math.sqrt(d) * 2).to(torch.bfloat16)
    bi, r, col = 2, 447, 2411
    r0, c0 = r - r % 32, col - col % 32
    i, j = r - r0, col - c0
    xt, wt = x[bi, r0:r0 + 32, :].contiguous(), w[:, c0:c0 + 32].contiguous()
    xr, wc = xt[i].double(), wt[:, j].double()
    rec = dict(elem=[bi, r, col], products=[float(v) for v in (xr * wc)])
    for fid, pl1 in (("HiFi4", True), ("HiFi4", False), ("HiFi3", True), ("HiFi2", True)):
        c = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                             fp32_dest_acc_en=True, packer_l1_acc=pl1)
        rows = []
        for k in range(1, d // 32 + 1):
            xd = ttnn.from_torch(xt[:, :32 * k].contiguous(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            wd = ttnn.from_torch(wt[:32 * k].contiguous(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            o = ttnn.to_torch(ttnn.matmul(xd, wd, compute_kernel_config=c, dtype=ttnn.float32)).double()
            ex = float((xr[:32 * k] * wc[:32 * k]).sum())
            ttnn.deallocate(xd), ttnn.deallocate(wd)
            rows.append([k, float(o[i, j]), ex, float(o[i, j]) - ex,
                         float((o - xt[:, :32 * k].double() @ wt[:32 * k].double()).abs().max())])
        rec[f"{fid}_{int(pl1)}"] = rows
        first = next((row for row in rows if abs(row[3]) > 0.25), None)
        print(fid, pl1, "first k off:", first, flush=True)
    open(a.out, "a").write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

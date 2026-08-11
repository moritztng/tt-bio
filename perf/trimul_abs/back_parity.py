#!/usr/bin/env python3
"""E2's correctness and rate: `reblock_permute_back` against `ttnn.permute` and the two transposes.

`torch.equal` at every (N, C) the trimul can reach at 512 aa and around it, not at one shape. The CB
ring-wrap failure this kernel family has produced before is silent and shape-dependent: it passes at
N=128 and N=256, where one group is the whole circular buffer, and produces garbage from N=512 up.
"""
import json, sys, time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tt_bio.reblock_permute as RB  # noqa: E402
from tt_bio.tenstorrent import COMPUTE_GRID_MAIN, get_device  # noqa: E402

dev = get_device()
OUT = {"grid": list(COMPUTE_GRID_MAIN), "rows": []}
NS = [int(v) for v in (sys.argv[1].split(",") if len(sys.argv) > 1
                       else "256,288,320,352,384,448,512,576,640".split(","))]
CS = [32, 64, 256]
MC = ttnn.DRAM_MEMORY_CONFIG


def rate(fn, x, bytes_rw, n=5):
    for _ in range(2):
        ttnn.deallocate(fn(x))
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = [fn(x) for _ in range(n)]
    ttnn.synchronize_device(dev)
    ms = (time.perf_counter() - t0) * 1e3 / n
    for o in outs:
        ttnn.deallocate(o)
    return ms, bytes_rw / (ms * 1e-3) / 1e9


torch.manual_seed(0)
for N in NS:
    if N % 32:
        continue
    for C in CS:
        ref_t = torch.randn(1, C, N, N, dtype=torch.float32).bfloat16()
        x = ttnn.from_torch(ref_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                            memory_config=MC)
        row = {"n": N, "c": C, "elig": RB.eligible_back(x, MC)}
        try:
            got = RB.reblock_permute_back(x, MC)
            want = ttnn.permute(x, (0, 2, 3, 1), memory_config=MC)
            gt, wt = ttnn.to_torch(got), ttnn.to_torch(want)
            row["equal_permute"] = bool(torch.equal(gt, wt))
            row["equal_ref"] = bool(torch.equal(gt, ref_t.permute(0, 2, 3, 1).contiguous()))
            row["shape"] = list(gt.shape)
            if not row["equal_permute"]:
                d = (gt.float() - wt.float()).abs()
                row["max_abs"] = float(d.max())
                row["frac_bad"] = float((d > 0).float().mean())
            ttnn.deallocate(got)
            ttnn.deallocate(want)
            # rate, only where it matters and only once per (N, C)
            b = 2 * C * N * N * 2
            row["kernel_ms"], row["kernel_gbs"] = rate(lambda t: RB.reblock_permute_back(t, MC), x, b)

            def two_transpose(t):
                a = ttnn.transpose(t, 1, 2, memory_config=MC)
                r = ttnn.transpose(a, 2, 3, memory_config=MC)
                ttnn.deallocate(a)
                return r
            row["stock_ms"], row["stock_gbs"] = rate(two_transpose, x, 2 * b)
            row["speedup"] = row["stock_ms"] / row["kernel_ms"]
        except Exception as e:                                              # noqa: BLE001
            row["err"] = str(e)[:300]
        ttnn.deallocate(x)
        OUT["rows"].append(row)
        print(f"N={N} C={C} elig={row.get('elig')} equal={row.get('equal_permute')} "
              f"ref={row.get('equal_ref')} kernel={row.get('kernel_ms', 0):.3f} ms "
              f"({row.get('kernel_gbs', 0):.0f} GB/s) stock={row.get('stock_ms', 0):.3f} "
              f"speedup={row.get('speedup', 0):.3f}x {row.get('err', '')}", flush=True)
        Path(sys.argv[2] if len(sys.argv) > 2 else "back_parity.json").write_text(
            json.dumps(OUT, indent=1))
print("rejects:", {f"{k[0]}{list(k[1])}": v for k, v in RB.REJECTS.items()})

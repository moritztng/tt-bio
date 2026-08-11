#!/usr/bin/env python3
"""E4's correctness and rate: the forward channel move after the reader batches its DRAM reads.

Ragged N is the case that matters here and the back direction does not have it: the forward kernel
serves 298 aa in production folds, where the last row-group is 10 real rows and 22 tile-padding rows
the writer must zero. The batched reader reserves the same fixed 32-tile window and still issues a
real read for the padding rows, so the group stays at 32 pushes -- but that is the claim being tested,
not an assumption.
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
                       else "256,288,298,320,352,384,448,512,576,640".split(","))]
CS = [32, 64, 256]


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
    for C in CS:
        ref_t = torch.randn(1, N, N, C, dtype=torch.float32).bfloat16()
        for mc, tag in ((ttnn.DRAM_MEMORY_CONFIG, "DRAM"), (ttnn.L1_MEMORY_CONFIG, "L1")):
            x = ttnn.from_torch(ref_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            row = {"n": N, "c": C, "dst": tag, "elig": RB.eligible(x, mc)}
            if not row["elig"]:
                OUT["rows"].append(row)
                ttnn.deallocate(x)
                continue
            try:
                got = RB.reblock_permute(x, mc)
                want = ttnn.permute(x, (0, 3, 1, 2), memory_config=mc)
                gt, wt = ttnn.to_torch(got), ttnn.to_torch(want)
                row["equal_permute"] = bool(torch.equal(gt, wt))
                row["equal_ref"] = bool(torch.equal(gt, ref_t.permute(0, 3, 1, 2).contiguous()))
                if not row["equal_permute"]:
                    d = (gt.float() - wt.float()).abs()
                    row["max_abs"] = float(d.max())
                    row["frac_bad"] = float((d > 0).float().mean())
                ttnn.deallocate(got)
                ttnn.deallocate(want)
                b = 2 * C * N * N * 2
                row["kernel_ms"], row["kernel_gbs"] = rate(lambda t: RB.reblock_permute(t, mc), x, b)
                row["stock_ms"], row["stock_gbs"] = rate(
                    lambda t: ttnn.permute(t, (0, 3, 1, 2), memory_config=mc), x, b)
                row["speedup"] = row["stock_ms"] / row["kernel_ms"]
            except Exception as e:                                          # noqa: BLE001
                row["err"] = str(e)[:300]
            ttnn.deallocate(x)
            OUT["rows"].append(row)
            print(f"N={N} C={C} ->{tag} equal={row.get('equal_permute')} ref={row.get('equal_ref')} "
                  f"kernel={row.get('kernel_ms', 0):.3f} ms ({row.get('kernel_gbs', 0)/2:.0f} GB/s "
                  f"each way) stock={row.get('stock_ms', 0):.3f} speedup={row.get('speedup', 0):.3f}x "
                  f"{row.get('err', '')}", flush=True)
            Path(sys.argv[2] if len(sys.argv) > 2 else "fwd_parity.json").write_text(
                json.dumps(OUT, indent=1))
print("rejects:", {f"{k[0]}{list(k[1])}": v for k, v in RB.REJECTS.items()})

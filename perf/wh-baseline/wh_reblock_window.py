"""Where does `reblock_permute`'s win actually start and stop on a 72-core Wormhole grid?

`eligible()` opens the L1 leg on `288 <= N <= 352` and the DRAM leg on `N >= 256`. Both edges were
fitted on Blackhole, and the docstring names the mechanism: work groups (Nt^2, Nt = N/32) against
cores. On 130 cores the collapse point is N=384 (144 groups, 1.11 waves). On 72 cores the same
wave count arrives at N~272, i.e. inside the shipped L1 window.

This sweeps N against the wheel's own `ttnn.permute` on both destination buffer types, checks
`torch.equal` at every point, and records what `eligible()` says so the gate's opinion and the
measurement sit side by side.

Run with TT_VISIBLE_DEVICES=<free umd id>; the visible chip re-indexes to device 0.
"""
import json, os, statistics, sys, time

import torch
import ttnn

from tt_bio import reblock_permute as rp

DEV = int(os.environ.get("PROBE_DEV", "0"))
WARMUP = int(os.environ.get("PROBE_WARMUP", "2"))
ITERS = int(os.environ.get("PROBE_ITERS", "10"))
BLOCKS = int(os.environ.get("PROBE_BLOCKS", "5"))
C = int(os.environ.get("PROBE_C", "128"))

NS = [224, 256, 272, 288, 320, 352, 384, 416, 448, 512]


def timed(fn, device):
    for _ in range(WARMUP):
        fn().deallocate()
    ttnn.synchronize_device(device)
    ms = []
    for _ in range(BLOCKS):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            # Free each output before the next call. Holding all ITERS of them alive is what an
            # earlier version of this script did, and on an L1 destination it OOMed the whole
            # sweep -- a harness artifact that looked exactly like a real L1 capacity finding.
            fn().deallocate()
        ttnn.synchronize_device(device)
        ms.append((time.perf_counter() - t0) * 1e3 / ITERS)
    return {"best": min(ms), "median": statistics.median(ms), "all": [round(x, 4) for x in ms]}


def main():
    device = ttnn.open_device(device_id=DEV)
    try:
        g = device.compute_with_storage_grid_size()
        cores = g.x * g.y
        out = {"grid": [g.x, g.y], "cores": cores, "C": C,
               "L1_N_MIN": rp.L1_N_MIN, "L1_N_MAX": rp.L1_N_MAX,
               "warmup": WARMUP, "iters": ITERS, "blocks": BLOCKS, "points": []}
        for bt_name, bt in (("DRAM", ttnn.BufferType.DRAM), ("L1", ttnn.BufferType.L1)):
            mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, bt)
            for N in NS:
                nt = (N + 31) // 32
                rec = {"N": N, "dest": bt_name, "groups": nt * nt,
                       "waves": round(nt * nt / cores, 3)}
                t = torch.randn((1, N, N, C), dtype=torch.float32).to(torch.bfloat16)
                x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=device,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
                try:
                    rec["eligible"] = bool(rp.eligible(x, mc))
                    a = rp.reblock_permute(x, mc, device)
                    b = ttnn.permute(x, (0, 3, 1, 2), memory_config=mc)
                    rec["bit_exact"] = bool(torch.equal(ttnn.to_torch(a), ttnn.to_torch(b)))
                    a.deallocate(); b.deallocate()
                    rec["kernel"] = timed(lambda: rp.reblock_permute(x, mc, device), device)
                    rec["ttnn"] = timed(
                        lambda: ttnn.permute(x, (0, 3, 1, 2), memory_config=mc), device)
                    rec["ratio_ttnn_over_kernel"] = round(
                        rec["ttnn"]["median"] / rec["kernel"]["median"], 4)
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                x.deallocate()
                out["points"].append(rec)
                print(json.dumps(rec), flush=True)
        print("RESULT " + json.dumps(out))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())

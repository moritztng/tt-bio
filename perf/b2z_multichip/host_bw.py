"""Host<->device bandwidth on one pinned chip, at the payloads a row-sharded z implies.

This prices the only chip-to-chip path that does not need tt-fabric: relay the shard through host
DRAM. The Galaxy reports pcie_width 1 on 28 of its 32 ASICs and 8 on four of them (UMD 5/13/21/29),
so the answer is expected to differ per chip and both cases are measured.
"""
import argparse, json, os, statistics, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--card", required=True)
ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--out", default=None)
a = ap.parse_args()
os.environ["TT_VISIBLE_DEVICES"] = a.card

import torch                                                      # noqa: E402
import ttnn                                                       # noqa: E402

dev = ttnn.open_device(device_id=0,
                       dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER))
rows = []
try:
    # z at 512 aa is (512,512,128) bf16 = 67.108864 MB; the k=2 shard is half of it.
    for label, shape in (("z_quarter", (1, 128, 512, 128)),
                         ("z_half_512aa", (1, 256, 512, 128)),
                         ("z_full_512aa", (1, 512, 512, 128))):
        t = torch.zeros(shape, dtype=torch.bfloat16)
        nbytes = t.numel() * 2
        up, down = [], []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            d = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.synchronize_device(dev)
            up.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            _ = ttnn.to_torch(d)
            down.append(time.perf_counter() - t0)
            ttnn.deallocate(d)
        u, w = statistics.median(up), statistics.median(down)
        rows.append({"label": label, "bytes": nbytes, "card": a.card,
                     "h2d_s": u, "d2h_s": w,
                     "h2d_GBps": nbytes / u / 1e9, "d2h_GBps": nbytes / w / 1e9})
        print(f"card {a.card:>3s} {label:14s} {nbytes/1e6:8.2f} MB  "
              f"H2D {u*1e3:8.2f} ms ({nbytes/u/1e9:5.2f} GB/s)  "
              f"D2H {w*1e3:8.2f} ms ({nbytes/w/1e9:5.2f} GB/s)", flush=True)
finally:
    ttnn.close_device(dev)
if a.out:
    open(a.out, "w").write(json.dumps(rows, indent=1))
print("HOSTBW-DONE", flush=True)

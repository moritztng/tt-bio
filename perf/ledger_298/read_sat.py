import json, sys, time, statistics as st
import torch, ttnn
from tt_bio.tenstorrent import get_device
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
dev = get_device()
def timed(fn, warm=3, pipe=4, reps=7):
    for _ in range(warm): fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        for _ in range(pipe): fn()
        ttnn.synchronize_device(dev); o.append((time.perf_counter() - t0) / pipe)
    return st.median(o)
out = []
for mb in (64, 80, 96, 112, 128):
    nrow = int(mb * 1e6 / 2) // 4096
    nbytes = nrow * 4096 * 2
    r = {"MB": round(nbytes / 1e6, 2)}
    try:
        xd = ttnn.from_torch(torch.randn(nrow, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=dev, memory_config=DRAM)
        r["read_GBs"] = round(nbytes / timed(lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=L1))) / 1e9, 1)
        r["dram2dram_rw_GBs"] = round(2 * nbytes / timed(lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=DRAM))) / 1e9, 1)
        ttnn.deallocate(xd)
    except Exception as e:
        r["err"] = str(e)[:90]
    out.append(r); print("  " + json.dumps(r), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=2)

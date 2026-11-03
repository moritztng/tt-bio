# DRAM write ceiling on this part: device fill (write only) and clone (read + write) of a [S, S, c] bf16 tile tensor.
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(ROOT / "perf")]
import ttnn
import tt_bio.tenstorrent as T
import clocksample
dev = T.get_device()


def timed(fn, reps=7):
    out = fn(); ttnn.synchronize_device(dev); out.deallocate()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); ttnn.synchronize_device(dev); ts.append(time.perf_counter() - t)
        out.deallocate()
    return sorted(ts)[len(ts) // 2] * 1e3


for S, c in ((1024, 320), (1024, 1280), (512, 1280)):
    shape = [1, S, S, c]
    nbytes = S * S * c * 2
    x = ttnn.full(shape, 1.0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    arms = {"full": lambda: ttnn.full(shape, 1.0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
            "clone": lambda: ttnn.clone(x)}
    for name, fn in list(arms.items()) * 2:
        with clocksample.during(period=0.5) as clk:
            ms = timed(fn)
        moved = nbytes * (2 if name == "clone" else 1)
        print(json.dumps({"S": S, "c": c, "arm": name, "ms": round(ms, 3), "MB": round(nbytes / 1e6),
                          "GBps_moved": round(moved / ms / 1e6, 1), "aiclk": clk.summary()}), flush=True)
    x.deallocate()

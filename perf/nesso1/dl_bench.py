"""Time the trunk's z readback at Nesso-1 sizes: ttnn.to_torch of a bf16 TILE tensor (what PairformerModule._to_torch
does today) against an on-device ROW_MAJOR conversion first. Prints seconds per variant and whether they are equal."""
import argparse
import json
import time

import torch

from tt_bio import tenstorrent
import ttnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[544, 256])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out")
    a = ap.parse_args()
    dev = tenstorrent.get_device()
    res = {}
    for n in a.n:
        x = torch.randn(1, n, n, 128)
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        ref = None
        for name in ("tile", "tile_f32", "rm_dev", "rm_dev_f32"):
            ts = []
            for _ in range(a.reps):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                src = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT) if name.startswith("rm") else t
                y = ttnn.to_torch(src)
                if name.endswith("f32"):
                    y = y.to(torch.float32)
                ts.append(time.perf_counter() - t0)
                if src is not t:
                    ttnn.deallocate(src)
            same = None if ref is None else bool(torch.equal(y.float(), ref))
            ref = y.float() if ref is None else ref
            res[f"{name}@{n}"] = {"s": sorted(ts), "equal_to_tile": same}
            print(n, name, [round(v, 4) for v in sorted(ts)], same, flush=True)
        ttnn.deallocate(t)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

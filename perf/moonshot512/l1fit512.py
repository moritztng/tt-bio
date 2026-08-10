#!/usr/bin/env python3
"""Does a 512 aa pair tensor fit an L1 copy, and what does the fallback cost.

`ceiling-298aa.md` §4 prices every non-arithmetic Pairformer op at the measured L1<->L1 clone
roof. That is legal at 298 aa: the pair tensor is 52.4 MB, so a clone needs 104.9 MB of the
chip's 160.8 MB. At 512 aa the same tensor is 134.2 MB and a clone needs 268.4 MB, which this
chip does not have. So the 512 aa ceiling cannot inherit that roof -- it needs to know, per
shape, whether the L1 route exists at all and what the DRAM route costs when it does not.

Sweeps the pair-tensor channel widths the trunk really uses at both sizes, records the L1 clone
where it fits and the exception verbatim where it does not, and takes the DRAM clone at every
shape so the refused ones have a priced fallback.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:<slug> \
      python3 perf/moonshot512/l1fit512.py --out perf/moonshot512/l1fit_qb2c1.json
"""
import argparse
import json
import statistics as st
import sys
import time

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/perf/", 1)[0])
from tt_bio.tenstorrent import get_device  # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG


def timed(dev, fn, warm=3, pipe=4, reps=5):
    """Identical protocol to perf/ceiling/k_sweep.py so the 320 aa rows are comparable."""
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3 / pipe)
    return st.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="320,512", help="padded token counts to sweep")
    ap.add_argument("--channels", default="32,64,128,256")
    args = ap.parse_args()

    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    cores = g.x * g.y
    try:
        from tt_bio.tenstorrent import _l1_bank_bytes
        l1_total = _l1_bank_bytes() * cores
    except Exception:                                                    # noqa: BLE001
        l1_total = None
    res = {"grid": [g.x, g.y], "cores": cores, "l1_total_bytes": l1_total, "rows": []}
    print(f"grid {g.x}x{g.y} = {cores} cores, L1 total "
          f"{'unknown' if l1_total is None else f'{l1_total/1e6:.1f} MB'}", flush=True)

    torch.manual_seed(0)
    for n in [int(x) for x in args.sizes.split(",")]:
        for c in [int(x) for x in args.channels.split(",")]:
            shape = (1, n, n, c)
            byt = 2 * n * n * c
            row = {"shape": list(shape), "MB": round(byt / 1e6, 2),
                   "two_live_MB": round(2 * byt / 1e6, 2)}
            src_l1 = None
            try:
                src_l1 = ttnn.from_torch(torch.randn(*shape) * 0.05, layout=ttnn.TILE_LAYOUT,
                                         device=dev, dtype=ttnn.bfloat16, memory_config=L1)
                ms = timed(dev, lambda: ttnn.deallocate(ttnn.clone(src_l1, memory_config=L1)))
                row["l1_fits"] = True
                row["l1_ms"] = round(ms, 5)
                row["l1_GBs_rw"] = round(2 * byt / 1e9 / (ms / 1e3), 1)
            except Exception as e:                                       # noqa: BLE001
                row["l1_fits"] = False
                row["l1_error"] = str(e)[:300]
            finally:
                if src_l1 is not None:
                    try:
                        ttnn.deallocate(src_l1)
                    except Exception:                                    # noqa: BLE001
                        pass

            src = ttnn.from_torch(torch.randn(*shape) * 0.05, layout=ttnn.TILE_LAYOUT,
                                 device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
            ms = timed(dev, lambda: ttnn.deallocate(ttnn.clone(src, memory_config=DRAM)))
            row["dram_ms"] = round(ms, 5)
            row["dram_GBs_rw"] = round(2 * byt / 1e9 / (ms / 1e3), 1)
            ttnn.deallocate(src)

            res["rows"].append(row)
            fit = f"L1 {row['l1_GBs_rw']:7.1f} GB/s" if row["l1_fits"] else "L1 REFUSED       "
            print(f"  {str(shape):22s} {byt/1e6:7.2f} MB (2 live {2*byt/1e6:7.2f})  "
                  f"{fit}   DRAM {row['dram_GBs_rw']:7.1f} GB/s", flush=True)

    fits = [r for r in res["rows"] if r["l1_fits"]]
    res["l1_copy_roof_GBs"] = max((r["l1_GBs_rw"] for r in fits), default=None)
    res["dram_copy_roof_GBs"] = max(r["dram_GBs_rw"] for r in res["rows"])
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nL1 copy roof (largest shape that fits)  {res['l1_copy_roof_GBs']} GB/s rw", flush=True)
    print(f"DRAM copy roof                          {res['dram_copy_roof_GBs']} GB/s rw", flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Where the 640 -> 768 aa TriangleMultiplication cliff lives, op by op, off-lattice.

`trimul_mm_size.py` exonerated the triangle matmul: it is 1.675 ms at 640 aa and 2.606 ms at 768
per group iteration, 10.2 % and 6.1 % of the module wall, and its efficiency IMPROVES across the
step. Everything else in the channel loop scales with N^2, and the module's non-matmul remainder
goes 29.57 -> 80.03 ms per call across the same step, 2.71x for a 1.44x growth in bytes. So one of
the layout moves is losing bandwidth, not doing more work.

Hypothesis: DRAM bank aliasing. An interleaved tile tensor round-robins pages over the part's DRAM
banks, so when the tile count per row is a multiple of the bank count every tile in a tile-COLUMN
lands on the same bank, and a column-strided reader gets one bank's bandwidth instead of all of
them. Nt is 16 at 512 aa, 20 at 640, 24 at 768 and 32 at 1024, and the measured TriMul efficiency
(s per (N/512)^3) is 23.90 / 20.49 / 30.73 / 35.18 -- bad exactly at the multiples of 8.

This sweeps every 32-aligned N from 512 to 1024 and times the four column-strided moves the channel
loop actually issues, at OpenDDE's own [1, 192, N, N] chunk shape. A spike at Nt = 16, 24, 32 and
nowhere else confirms aliasing; a smooth curve refutes it and the cliff is somewhere else.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WARM, REPS = 1, 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(str(n) for n in range(512, 1025, 32)))
    ap.add_argument("--channels", type=int, default=192)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch, ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.reblock_permute as RB
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    DRAM = ttnn.DRAM_MEMORY_CONFIG
    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "grid": [g.x, g.y], "channels": a.channels,
           "dram_banks": int(dev.num_dram_channels()) if hasattr(dev, "num_dram_channels") else None,
           "rows": []}
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}), flush=True)

    def bench(fn):
        ts = []
        for _ in range(WARM):
            o = fn()
            if o is not None:
                ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        for _ in range(REPS):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = fn()
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
            if o is not None:
                ttnn.deallocate(o)
        return round(st.median(ts), 4)

    C = a.channels
    for N in [int(s) for s in a.sizes.split(",")]:
        Nt = -(-N // 32)
        row = {"N": N, "Nt": Nt, "Nt_mod8": Nt % 8}
        # chunk in channel-last form, as the in-projection produces it
        try:
            cl = ttnn.from_torch(torch.zeros([1, N, N, C], dtype=torch.bfloat16),
                                 layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            cf = ttnn.from_torch(torch.zeros([1, C, N, N], dtype=torch.bfloat16),
                                 layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            row["transpose_wh_ms"] = bench(lambda: ttnn.transpose(cf, -2, -1, memory_config=DRAM))
            row["permute_0312_ms"] = bench(lambda: ttnn.permute(cl, (0, 3, 1, 2), memory_config=DRAM))
            row["reallocate_ms"] = bench(lambda: ttnn.reallocate(
                ttnn.clone(cf, memory_config=DRAM)))
            row["clone_ms"] = bench(lambda: ttnn.clone(cf, memory_config=DRAM))
            row["eligible_back"] = bool(RB.eligible_back(cf, DRAM))
            if row["eligible_back"]:
                row["reblock_back_ms"] = bench(
                    lambda: RB.reblock_permute_back(cf, DRAM))
            row["eligible_fwd"] = bool(RB.eligible(cl, DRAM))
            if row["eligible_fwd"]:
                row["reblock_fwd_ms"] = bench(lambda: RB.reblock_permute(cl, DRAM))
            ttnn.deallocate(cl); ttnn.deallocate(cf)
        except Exception as exc:                                       # noqa: BLE001
            row["error"] = str(exc).strip().split("\n")[-1][:160]
        # bytes-normalised: every op above moves 2 * C * N^2 * 2 bytes
        gb = 2 * C * N * N * 2 / 1e9
        for k in [k for k in list(row) if k.endswith("_ms")]:
            row[k.replace("_ms", "_gbps")] = round(gb / (row[k] / 1e3), 1)
        print(json.dumps(row), flush=True)
        res["rows"].append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()

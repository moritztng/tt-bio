#!/usr/bin/env python3
"""T2 probe 2 — the permute mechanism, and the two-NOC cross-check.

H  Why does the trimul channel-move permute run at ~139 GB/s when a clone of the same bytes in
   the same buffers runs at ~850?  The kernel cache says ttnn has BOTH a tiled permute writer
   (writer_permute_interleaved_tiled_*) and a ROW-MAJOR one (writer_permute_interleaved_rm_*).
   Hypothesis: the rank-4 channel move takes the row-major route, i.e. untilize -> row-major
   move -> tilize, and the layout conversion is what costs, not the data movement.
   Falsifier: an explicit to_layout(ROW_MAJOR) + to_layout(TILE) round trip on the same tensor
   costs far LESS than the permute.
   Controls: (a) permute(0,2,1,3), a whole-tile reorder with no sub-tile shuffle; (b) the same
   permute with DRAM operands, to test buffer-type independence; (c) transpose(-2,-1), the
   tile-local op the production code already uses on the DRAM path.

N  Two-NOC cross-check.  Every writer kernel in this ttnn build is compiled onto BRISC and every
   reader onto NCRISC.  If reads and writes therefore ride different NOCs, a DRAM->DRAM clone
   should move read+write bytes faster than either directional roof alone.
"""
import argparse
import json
import statistics as st
import time

import torch
import ttnn

from tt_bio.tenstorrent import get_device, COMPUTE_GRID_MAIN

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def timed(dev, fn, warm=4, pipe=5, reps=7):
    for _ in range(warm):
        r = fn()
        del r
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        keep = [fn() for _ in range(pipe)]
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / pipe)
        del keep
    return st.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = get_device()
    gx, gy = COMPUTE_GRID_MAIN
    res = {"grid": f"{gx}x{gy}"}

    N, C = 320, 32
    nb = N * N * C * 2                      # one chunk, bytes
    moved = 2 * nb                          # read + write

    print("=== H permute mechanism, [1,320,320,32] bf16 ===", flush=True)
    H = {}
    for buf_lbl, buf in (("L1", L1), ("DRAM", DRAM)):
        x = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=buf)
        legs = {
            "permute_0312": lambda: ttnn.permute(x, (0, 3, 1, 2), memory_config=buf),
            "permute_0321": lambda: ttnn.permute(x, (0, 3, 2, 1), memory_config=buf),
            "permute_0213_wholetile": lambda: ttnn.permute(x, (0, 2, 1, 3), memory_config=buf),
            "transpose_m2_m1": lambda: ttnn.transpose(x, -2, -1, memory_config=buf),
            "clone": lambda: ttnn.clone(x, memory_config=buf),
            "typecast_same": lambda: ttnn.typecast(x, ttnn.bfloat16, memory_config=buf),
        }
        for lbl, fn in legs.items():
            try:
                s = timed(dev, fn)
                H[f"{buf_lbl}_{lbl}"] = {"us": round(s * 1e6, 1),
                                         "gbps": round(moved / s / 1e9, 1)}
                print(f"  {buf_lbl:4s} {lbl:24s} {s*1e6:8.1f} us  {moved/s/1e9:7.1f} GB/s",
                      flush=True)
            except Exception as e:
                H[f"{buf_lbl}_{lbl}"] = {"err": str(e)[:90]}
                print(f"  {buf_lbl} {lbl} ERR {str(e)[:90]}", flush=True)
        # the layout round trip: untilize then re-tilize, same tensor
        try:
            rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.synchronize_device(dev)
            s_rm = timed(dev, lambda: ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), warm=3, pipe=4)
            s_ti = timed(dev, lambda: ttnn.to_layout(rm, ttnn.TILE_LAYOUT), warm=3, pipe=4)
            H[f"{buf_lbl}_untilize"] = {"us": round(s_rm * 1e6, 1),
                                        "gbps": round(moved / s_rm / 1e9, 1)}
            H[f"{buf_lbl}_tilize"] = {"us": round(s_ti * 1e6, 1),
                                      "gbps": round(moved / s_ti / 1e9, 1)}
            H[f"{buf_lbl}_roundtrip_us"] = round((s_rm + s_ti) * 1e6, 1)
            print(f"  {buf_lbl:4s} untilize {s_rm*1e6:8.1f} us | tilize {s_ti*1e6:8.1f} us | "
                  f"round trip {(s_rm+s_ti)*1e6:8.1f} us", flush=True)
            ttnn.deallocate(rm)
        except Exception as e:
            H[f"{buf_lbl}_roundtrip"] = {"err": str(e)[:90]}
            print(f"  {buf_lbl} roundtrip ERR {str(e)[:90]}", flush=True)
        ttnn.deallocate(x)
    res["H_permute"] = H

    print("=== N two-NOC cross-check: DRAM->DRAM vs directional ===", flush=True)
    Nn = []
    for mb in (8, 16, 32):
        rows = int(mb * 1e6 / 2) // 4096
        b = rows * 4096 * 2
        rec = {"MB": round(b / 1e6, 2)}
        try:
            xd = ttnn.from_torch(torch.randn(rows, 4096), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
            s = timed(dev, lambda: ttnn.clone(xd, memory_config=DRAM), warm=2, pipe=3, reps=5)
            rec["d2d_rw_gbps"] = round(2 * b / s / 1e9, 1)
            rec["d2d_perdir_gbps"] = round(b / s / 1e9, 1)
            rec["ms"] = round(s * 1e3, 4)
            ttnn.deallocate(xd)
        except Exception as e:
            rec["err"] = str(e)[:80]
        Nn.append(rec)
        print("  " + json.dumps(rec), flush=True)
    res["N_d2d"] = Nn

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("WROTE " + args.out, flush=True)


if __name__ == "__main__":
    main()

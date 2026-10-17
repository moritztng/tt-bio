#!/usr/bin/env python3
"""reblock_permute_back at the trimul's DRAM sizes: parity against torch, then ms/call per stage ablation.

The fold census puts the move at 2.2 ms a call on Wormhole at 512 c256 (wtm Mops), ~2.4x its DRAM floor
(read + write 268 MB at 288 GB/s). `--abl` times the kernel with the DRAM read (1), the L1 gather (2) or
the DRAM write (4) skipped, so the remaining time says which stage bounds it, then BACK_SPLIT
(the writer reads one group ahead on its own NOC) with its own parity check. Bench only: ablated outputs are garbage.
Run under `timeout`; the runner passes --out.
"""
import argparse, json, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="384,512")
ap.add_argument("--c", type=int, default=256)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--abl", default="1,2,4,3,6", help="comma list of BACK_ABL masks timed after the full kernel")
ap.add_argument("--walks", default="", help="comma list of RB.WALK modes (block,stride,rotate) timed with parity")
ap.add_argument("--ref", action="store_true", help="also time ttnn.clone and ttnn.permute of the same tensor (DRAM yardsticks)")
ap.add_argument("--out")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, reblock_permute as RB

dev = T.get_device()


def timed(f, reps):
    f(); ttnn.synchronize_device(dev)
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(); ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ms), 4)


gen = torch.Generator().manual_seed(0)
for n in (int(v) for v in A.ns.split(",")):
    xt = torch.randn(1, A.c, n, n, generator=gen).to(torch.bfloat16)
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, n, n, A.c]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                         ttnn.DRAM_MEMORY_CONFIG)
    RB.BACK_ABL = 0
    run = lambda: RB.reblock_permute_back(x, ttnn.DRAM_MEMORY_CONFIG, out=out)
    run(); ttnn.synchronize_device(dev)
    exact = bool(torch.equal(ttnn.to_torch(out), xt.permute(0, 2, 3, 1)))
    floor_ms = 2 * xt.numel() * 2 / 288e9 * 1e3
    rec = {"n": n, "c": A.c, "exact": exact, "full_ms": timed(run, A.reps), "wh_floor_ms": round(floor_ms, 3)}
    for m in (int(v) for v in A.abl.split(",") if v):
        RB.BACK_ABL = m
        rec[f"abl{m}_ms"] = timed(run, A.reps)
    RB.BACK_ABL = 0
    # BACK_SPLIT: the writer stages the DRAM read one group ahead (RPB_SPLIT). Parity first, then time.
    RB.BACK_SPLIT = True
    run(); ttnn.synchronize_device(dev)      # `out` holds the last ablation's garbage, so a skipped page shows
    rec["split_exact"] = bool(torch.equal(ttnn.to_torch(out), xt.permute(0, 2, 3, 1)))
    rec["split_ms"] = timed(run, A.reps)
    RB.BACK_SPLIT = False
    rec["full_ms_again"] = timed(run, A.reps)
    # The walk decides which DRAM banks are live together (reblock_permute.WALK); bit-identical by construction.
    for w in (v for v in A.walks.split(",") if v):
        RB.WALK = w
        run(); ttnn.synchronize_device(dev)
        rec[f"walk_{w}_exact"] = bool(torch.equal(ttnn.to_torch(out), xt.permute(0, 2, 3, 1)))
        rec[f"walk_{w}_ms"] = timed(run, A.reps)
    RB.WALK = "block"
    if A.ref:
        # What a stock read+write of the same 134 MB reaches on this chip: the practical roof for the move.
        rec["clone_ms"] = timed(lambda: ttnn.deallocate(ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)), A.reps)
        rec["permute_ms"] = timed(lambda: ttnn.deallocate(ttnn.permute(x, (0, 2, 3, 1))), A.reps)
        rec["clone_gbs"] = round(2 * xt.numel() * 2 / rec["clone_ms"] / 1e6, 1)
        rec["full_gbs"] = round(2 * xt.numel() * 2 / rec["full_ms"] / 1e6, 1)
    print(json.dumps(rec), flush=True)
    if A.out:
        with open(A.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    ttnn.deallocate(x); ttnn.deallocate(out)

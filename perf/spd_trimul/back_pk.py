"""permute_back_pk vs reblock_permute_back at the trimul's DRAM sizes: parity against torch, ms/call, and
permute_back_pk's stage ablations (1 no read, 2 no tilize/transpose, 4 no write). Run under `timeout`."""
import argparse, json, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="512")
ap.add_argument("--c", type=int, default=256)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--abl", default="1,2,4,6")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, reblock_permute as RB, permute_back_pk as PK

dev = T.get_device()
grid = tuple(T.COMPUTE_GRID_MAIN)


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
    ref = xt.permute(0, 2, 3, 1)
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    mk = lambda: ttnn.allocate_tensor_on_device(ttnn.Shape([1, n, n, A.c]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                                ttnn.DRAM_MEMORY_CONFIG)
    o1, o2 = mk(), mk()
    old = lambda: RB.reblock_permute_back(x, ttnn.DRAM_MEMORY_CONFIG, out=o1)
    new = lambda: PK.permute_back_pk(x, o2, grid)
    old(); new(); ttnn.synchronize_device(dev)
    got = ttnn.to_torch(o2)
    rec = {"n": n, "c": A.c, "grid": grid, "old_exact": bool(torch.equal(ttnn.to_torch(o1), ref)),
           "pk_exact": bool(torch.equal(got, ref)),
           "pk_mismatch": int((got != ref).sum()), "old_ms": timed(old, A.reps), "pk_ms": timed(new, A.reps),
           "wh_floor_ms": round(2 * xt.numel() * 2 / 288e9 * 1e3, 3)}
    # half the channels into the second half of a 2c output (the chunked caller's c_off)
    for m in (int(v) for v in A.abl.split(",") if v):
        PK.ABL = m
        rec[f"pk_abl{m}_ms"] = timed(new, A.reps)
    PK.ABL = 0
    print(json.dumps(rec), flush=True)

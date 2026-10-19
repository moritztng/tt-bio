"""permute_back_pk vs reblock_permute_back at the trimul's DRAM sizes: parity against torch, ms/call, and
permute_back_pk's stage ablations (1 no read, 2 no tilize/transpose, 4 no write). Run under `timeout`."""
import argparse, json, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="512")
ap.add_argument("--c", type=int, default=256)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--abl", default="1,2,4,8")
# The fold's pattern: every call on freshly allocated x and out (new addresses), channel halves c_off 0 / C of a 2C
# output, x in DRAM or L1 per call. Exactness of every call; a hang names the call it stopped at.
ap.add_argument("--fresh", type=int, default=0)
# Back-to-back pk calls, no sync between, c_off alternating, the old kernel between every pk call so the device
# switches programs as in a fold. Synced every 200; exactness checked at the end. For an intermittent hang.
ap.add_argument("--stress", type=int, default=0)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, reblock_permute as RB, permute_back_pk as PK

dev = T.get_device()
RB._PBPK = False      # `old` is reblock_permute_back's own kernel, not the pk route it takes on Wormhole
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
    old(); ttnn.synchronize_device(dev)
    print(json.dumps({"stage": "old ok"}), flush=True)
    # Risk order, one stage per launch, so a hang names its stage: untilize only, + tilize, full.
    for m in (2, 8):
        PK.ABL = m
        new(); ttnn.synchronize_device(dev)
        print(json.dumps({"stage": f"pk abl{m} ok"}), flush=True)
    PK.ABL = 0
    new(); ttnn.synchronize_device(dev)
    print(json.dumps({"stage": "pk full ok"}), flush=True)
    got = ttnn.to_torch(o2)
    rec = {"n": n, "c": A.c, "grid": grid, "old_exact": bool(torch.equal(ttnn.to_torch(o1), ref)),
           "pk_exact": bool(torch.equal(got, ref)),
           "pk_mismatch": int((got != ref).sum()), "old_ms": timed(old, A.reps), "pk_ms": timed(new, A.reps),
           "wh_floor_ms": round(2 * xt.numel() * 2 / 288e9 * 1e3, 3)}
    # c channels into the second half of a 2c output (the chunked caller's c_off)
    o3 = ttnn.allocate_tensor_on_device(ttnn.Shape([1, n, n, 2 * A.c]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                        ttnn.DRAM_MEMORY_CONFIG)
    PK.permute_back_pk(x, o3, grid, c_off=A.c); ttnn.synchronize_device(dev)
    rec["pk_coff_exact"] = bool(torch.equal(ttnn.to_torch(o3)[..., A.c:], ref))
    ttnn.deallocate(o3)
    for m in (int(v) for v in A.abl.split(",") if v):
        PK.ABL = m
        rec[f"pk_abl{m}_ms"] = timed(new, A.reps)
    PK.ABL = 0
    print(json.dumps(rec), flush=True)
    bad, keep = 0, []
    for i in range(A.fresh):
        mc = ttnn.L1_MEMORY_CONFIG if (i % 3 == 2 and xt.numel() * 2 < 40e6) else ttnn.DRAM_MEMORY_CONFIG
        keep.append(ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32 * (1 + i % 7), 1024]), ttnn.bfloat16,
                                                   ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG))  # shift addresses
        xi = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        oi = ttnn.allocate_tensor_on_device(ttnn.Shape([1, n, n, 2 * A.c]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                            ttnn.DRAM_MEMORY_CONFIG)
        co = A.c * (i % 2)
        PK.permute_back_pk(xi, oi, grid, c_off=co); ttnn.synchronize_device(dev)
        bad += not torch.equal(ttnn.to_torch(oi)[..., co:co + A.c], ref)
        ttnn.deallocate(xi); ttnn.deallocate(oi)
        if len(keep) > 5:
            ttnn.deallocate(keep.pop(0))
        print(json.dumps({"fresh": i, "x": "L1" if mc == ttnn.L1_MEMORY_CONFIG else "DRAM", "bad": bad}), flush=True)
    for t in keep:
        ttnn.deallocate(t)
    if A.stress:
        o4 = ttnn.allocate_tensor_on_device(ttnn.Shape([1, n, n, 2 * A.c]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                            ttnn.DRAM_MEMORY_CONFIG)
        t0 = time.perf_counter()
        for i in range(A.stress):
            PK.permute_back_pk(x, o4, grid, c_off=A.c * (i % 2))
            old()
            if i % 200 == 199:
                ttnn.synchronize_device(dev)
                print(json.dumps({"stress": i + 1, "s": round(time.perf_counter() - t0, 1)}), flush=True)
        ttnn.synchronize_device(dev)
        got = ttnn.to_torch(o4)
        print(json.dumps({"stress_done": A.stress, "lo_exact": bool(torch.equal(got[..., :A.c], ref)),
                          "hi_exact": bool(torch.equal(got[..., A.c:], ref))}), flush=True)
        ttnn.deallocate(o4)

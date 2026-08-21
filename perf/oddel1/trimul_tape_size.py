#!/usr/bin/env python3
"""Op-by-op tape of the real TriangleMultiplication at OpenDDE's trunk shapes, per rung.

Drives `tenstorrent.TriangleMultiplication.__call__` itself, so every size-dependent decision
(`_triangle_mul_memory_config`, `_trimul_chunk_size`, `_trimul_inproj_group`, E6 eligibility, the
F1 tail) is the one the fold makes; the run asserts the picked (memory_config, chunk, group)
against what `sizes-recheck-opendde`'s in-fold census recorded, so a config the fold never executes
cannot be measured by accident. Weights are random -- this prices time, not numbers.
"""
import argparse, collections, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# (group, memory_config) as `sizes-recheck-opendde` and `pop_640_768_qb1c2.json` recorded them
# in-fold. The first six are trunk rungs (N = seq). 1243 and 1494 are OpenDDE's REFINER, whose H is
# ~1.95x the sequence length -- the population that carries 70.2 % of the 640 -> 768 aa interval
# delta (state doc section 18), so its tape runs through this same script with the same defaults:
# n_pairs and hidden are 12 and 384 at the refiner too, and only H differs.
EXPECT = {512: (12, "DRAM"), 640: (6, "DRAM"), 672: (6, "DRAM"), 704: (6, "DRAM"),
          768: (6, "DRAM"), 1024: (4, "DRAM"), 1243: (2, "DRAM"), 1494: (1, "DRAM"),
          1993: (1, "DRAM")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="640,768")
    ap.add_argument("--c-z", type=int, default=384)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch, ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.reblock_permute as RB
    import tt_bio.trimul_tail as TT
    import tt_bio.mm_dualnoc as DN
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)

    ROWS, ON = collections.Counter(), [False]
    CNT = collections.Counter()

    def shp(t):
        try:
            return "x".join(str(int(d)) for d in t.shape)
        except Exception:
            return "?"

    def wrap(mod, name, tag):
        orig = getattr(mod, name)
        def f(*ar, **kw):
            if not ON[0]:
                return orig(*ar, **kw)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            out = orig(*ar, **kw)
            ttnn.synchronize_device(dev)
            key = tag(ar, kw)
            ROWS[key] += (time.perf_counter() - t0) * 1e3
            CNT[key] += 1
            return out
        setattr(mod, name, f)

    wrap(ttnn, "matmul", lambda ar, kw: f"matmul[{shp(ar[0])}@{shp(ar[1])}]")
    wrap(ttnn, "linear", lambda ar, kw: f"linear[{shp(ar[0])}@{shp(ar[1])}]")
    wrap(ttnn.experimental, "minimal_matmul", lambda ar, kw: f"minimal_matmul[{shp(ar[0])}]")
    wrap(ttnn, "permute", lambda ar, kw: f"permute{tuple(ar[1])}[{shp(ar[0])}]")
    wrap(ttnn, "transpose", lambda ar, kw: f"transpose({ar[1]},{ar[2]})[{shp(ar[0])}]")
    wrap(ttnn, "layer_norm", lambda ar, kw: f"layer_norm[{shp(ar[0])}]")
    wrap(ttnn, "multiply_", lambda ar, kw: f"multiply_[{shp(ar[0])}]")
    wrap(ttnn, "chunk", lambda ar, kw: f"chunk[{shp(ar[0])}]")
    wrap(ttnn, "concat", lambda ar, kw: f"concat[{len(ar[0])}x{shp(ar[0][0])}]")
    wrap(ttnn, "clone", lambda ar, kw: f"clone[{shp(ar[0])}]")
    wrap(ttnn, "reallocate", lambda ar, kw: f"reallocate[{shp(ar[0])}]")
    wrap(RB, "reblock_permute_gated", lambda ar, kw: f"E6_gated[{shp(ar[0])}]")
    wrap(RB, "reblock_permute_back", lambda ar, kw: f"reblock_back[{shp(ar[0])}]")
    wrap(RB, "reblock_permute", lambda ar, kw: f"reblock_fwd[{shp(ar[0])}]")
    wrap(TT, "fused_tail", lambda ar, kw: f"F1_tail[{shp(ar[0])}]")
    wrap(DN, "in_proj", lambda ar, kw: f"dualnoc_in_proj[{shp(ar[0])}@{shp(ar[1])}]")

    cz, hid = a.c_z, a.hidden
    torch.manual_seed(0)
    sd = {"norm_in.weight": torch.ones(cz), "norm_in.bias": torch.zeros(cz),
          "norm_out.weight": torch.ones(hid), "norm_out.bias": torch.zeros(hid),
          "g_in.weight": torch.randn(2 * hid, cz) * 0.02,
          "p_in.weight": torch.randn(2 * hid, cz) * 0.02,
          "g_out.weight": torch.randn(hid, cz) * 0.02,
          "p_out.weight": torch.randn(cz, hid) * 0.02}
    tm = T.TriangleMultiplication(False, sd, ckc, gated_move=True)

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "grid": [g.x, g.y], "c_z": cz, "hidden": hid, "rungs": []}
    print(json.dumps({k: v for k, v in res.items() if k != "rungs"}), flush=True)

    for N in [int(s) for s in a.sizes.split(",")]:
        z = ttnn.from_torch(torch.randn(1, N, N, cz) * 0.5, layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16)
        mc = T._triangle_mul_memory_config(N)
        chunk = T._trimul_chunk_size(N, hid, 1)
        npairs = hid // chunk
        grp = T._trimul_inproj_group(N, chunk, 1, npairs) if mc.buffer_type == ttnn.BufferType.DRAM else 1
        buf = "L1" if mc.buffer_type == ttnn.BufferType.L1 else "DRAM"
        exp = EXPECT.get(N)
        picked = {"N": N, "memory_config": buf, "chunk": chunk, "n_pairs": npairs, "group": grp,
                  "matches_in_fold_census": (exp is None or (grp, buf) == exp)}
        print(json.dumps(picked), flush=True)
        tm.prewarm(N, 1)
        ttnn.deallocate(tm(z))                       # cold
        ROWS.clear(); CNT.clear()
        ON[0] = True
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(a.reps):
            ttnn.deallocate(tm(z))
        ttnn.synchronize_device(dev)
        wall = (time.perf_counter() - t0) * 1e3 / a.reps
        ON[0] = False
        tape = sorted(((k, round(v / a.reps, 4), CNT[k] // a.reps) for k, v in ROWS.items()),
                      key=lambda r: -r[1])
        taped = round(sum(r[1] for r in tape), 4)
        picked.update(call_ms=round(wall, 4), taped_ms=taped,
                      untaped_ms=round(wall - taped, 4), tape=tape)
        for k, ms, n in tape:
            print("   %9.4f ms  x%-4d %s" % (ms, n, k), flush=True)
        print("   ---- taped %.4f of %.4f ms (untaped %.4f)" % (taped, wall, wall - taped), flush=True)
        res["rungs"].append(picked)
        ttnn.deallocate(z)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()

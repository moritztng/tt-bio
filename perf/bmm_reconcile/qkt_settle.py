#!/usr/bin/env python3
"""E7 — settle the q@k^T disagreement between G1 and E1, one card, one process, alternating arms.

G1 (`wk/perfwar-batched-matmul-config`) declines every `q @ k^T` on an `n_tiles > 4` gate, on a
measured 1.10x at Nt=10 and a 0.65x regression at Nt=19. E1 (`wk/perfwar-of3-matmul-sites`) applies
`q @ k^T` and measured 1.83x on the OpenFold3 trunk triangle attention. Both were taken on qb1 at
0.67.4 but on different cards AND on different shapes: G1's Nt=10 point is the DiT site (B=16),
E1's is the trunk site (B=1192). This script runs both legs' chooser on all three shapes in one
process on one card, so the comparison is like-for-like.

Both chooser functions below are copied verbatim from their branches; nothing is re-derived.
Timing syncs the device immediately before the clock starts and immediately before it stops
(WARROOM 2.4), and the arms are interleaved round-robin so host drift cannot land on one arm.
"""
import argparse, json, time
import torch, ttnn
from tt_bio.tenstorrent import get_device
import tt_bio.tenstorrent as T

DRAM = ttnn.DRAM_MEMORY_CONFIG


# ---------------------------------------------------------------- G1's chooser (cac0eead)
_TILE_READ_PER_TILE_MAC = 3.0  # G1's cost weight; see _batched_matmul_config on that branch


def g1_config(batch, m_tiles, k_tiles, n_tiles, elem_bytes, grid):
    gx, gy = grid
    cores = gx * gy
    if batch < 2 or n_tiles > 4 or batch * m_tiles < cores:
        return None
    block_w = 2 if k_tiles % 2 == 0 else 1
    l1 = int(ttnn.get_max_worker_l1_unreserved_size())
    tile, acc_tile = 1024 * elem_bytes, 4096
    best = ()
    for p in range(1, m_tiles + 1):
        if m_tiles % p or (p != m_tiles and batch * m_tiles // p > cores):
            continue
        if 2 * (p + n_tiles) * block_w * tile + p * n_tiles * (tile + acc_tile) > l1:
            continue
        blocks = batch * m_tiles // p
        reads = batch * m_tiles * k_tiles + blocks * k_tiles * n_tiles
        cost = max(_TILE_READ_PER_TILE_MAC * reads, -(-blocks // cores) * p * n_tiles * k_tiles)
        if not best or cost < best[0]:
            best = (cost, p)
    if not best:
        return None
    per_core_M = best[1]
    sub_w = max(w for w in range(1, min(4, n_tiles) + 1) if n_tiles % w == 0)
    sub_h = max(h for h in range(1, min(4 // sub_w, per_core_M) + 1) if per_core_M % h == 0)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=block_w,
        out_subblock_h=sub_h, out_subblock_w=sub_w,
        per_core_M=per_core_M, per_core_N=n_tiles)


# ---------------------------------------------------------------- E1's chooser (d8073b6b)
_MM_BLOCK_L1_BUDGET = 700 * 1024


def _out_subblock(per_core_M, per_core_N):
    best = (1, 1)
    for h in range(1, per_core_M + 1):
        if per_core_M % h:
            continue
        for w in range(1, per_core_N + 1):
            if per_core_N % w or h * w > 4:
                continue
            if h * w > best[0] * best[1]:
                best = (h, w)
    return best


def e1_config(batch, m_tiles, k_tiles, n_tiles, elem_bytes, grid):
    gx, gy = grid
    cores = gx * gy
    if batch < 2:
        return None
    per_core_M = m_tiles
    for d in range(1, m_tiles + 1):
        if m_tiles % d == 0 and batch * (m_tiles // d) <= cores:
            per_core_M = d
            break
    if per_core_M * n_tiles * 6 * 1024 > _MM_BLOCK_L1_BUDGET:
        return None
    h, w = _out_subblock(per_core_M, n_tiles)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=(2 if k_tiles > 2 and k_tiles % 2 == 0 else 1),
        out_subblock_h=h, out_subblock_w=w,
        per_core_M=per_core_M, per_core_N=n_tiles)


def manual_config(per_core_M, block_w, n_tiles, grid):
    h, w = _out_subblock(per_core_M, n_tiles)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=block_w,
        out_subblock_h=h, out_subblock_w=w, per_core_M=per_core_M, per_core_N=n_tiles)


def cfg_repr(c):
    if c is None:
        return "declined"
    return (f"per_core_M={c.per_core_M} per_core_N={c.per_core_N} "
            f"in0_block_w={c.in0_block_w} subblock={c.out_subblock_h}x{c.out_subblock_w}")


# ---------------------------------------------------------------- shapes
# (name, a_shape, b_shape, dtype, calls/fold at 298 aa, note)
def shapes():
    return [
        ("trunk_triatt_qkT", (298, 4, 298, 32), (298, 4, 32, 298), ttnn.bfloat16, 488,
         "tenstorrent.py:259 fp32_softmax q@kT -- E1 applies (1.83x), G1 declines (Nt=10)"),
        ("dit_qkT", (1, 16, 320, 64), (1, 16, 64, 320), ttnn.float32, 4800,
         "openfold3_diffusion_transformer.py:184 -- G1's Nt=10 1.10x point"),
        ("dit580_qkT", (1, 16, 608, 64), (1, 16, 64, 608), ttnn.bfloat16, 0,
         "580 aa DiT q@kT -- G1's Nt=19 0.65x point, not issued by the 298 aa fold"),
        ("trunk_triatt_av", (298, 4, 298, 298), (298, 4, 298, 32), ttnn.bfloat16, 488,
         "tenstorrent.py:272 attn@v -- both legs apply, control"),
    ]


def med(xs):
    return sorted(xs)[len(xs) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pipe", type=int, default=4)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--sweep", action="store_true", help="also sweep per_core_M x in0_block_w")
    a = ap.parse_args()

    dev = get_device()
    T._configure_active_compute_grid(dev)
    grid = T.COMPUTE_GRID_MAIN
    cores = grid[0] * grid[1]
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    out = {"grid": list(grid), "cores": cores,
           "l1_unreserved": int(ttnn.get_max_worker_l1_unreserved_size()), "shapes": {}}
    print(f"grid {grid} = {cores} cores, L1 unreserved {out['l1_unreserved']}")

    for name, ash, bsh, dt, calls, note in shapes():
        ta = torch.randn(*ash) * 0.1
        tb = torch.randn(*bsh) * 0.1
        A = ttnn.from_torch(ta, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt, memory_config=DRAM)
        B = ttnn.from_torch(tb, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt, memory_config=DRAM)
        pa, pb = list(A.padded_shape), list(B.padded_shape)
        batch = 1
        for d in pa[:-2]:
            batch *= int(d)
        Mt, Kt, Nt = int(pa[-2]) // 32, int(pa[-1]) // 32, int(pb[-1]) // 32
        eb = 4 if dt == ttnn.float32 else 2
        arms = {"ttnn": None,
                "G1": g1_config(batch, Mt, Kt, Nt, eb, grid),
                "E1": e1_config(batch, Mt, Kt, Nt, eb, grid)}
        rec = {"note": note, "a": list(ash), "b": list(bsh), "padded_a": [int(x) for x in pa],
               "padded_b": [int(x) for x in pb], "dtype": str(dt), "calls_per_fold": calls,
               "batch": batch, "Mt": Mt, "Kt": Kt, "Nt": Nt,
               "configs": {k: cfg_repr(v) for k, v in arms.items()}, "ms": {}, "bit_exact": {}}
        print(f"\n== {name}  B={batch} Mt/Kt/Nt={Mt}/{Kt}/{Nt} {dt}")
        for k, v in arms.items():
            print(f"   {k:4s} {cfg_repr(v)}")

        def run(c):
            kw = {"compute_kernel_config": ckc}
            if c is not None:
                kw["program_config"] = c
            return ttnn.matmul(A, B, **kw)

        # parity: every arm against the plain call
        ref = ttnn.to_torch(run(None))
        for k, v in arms.items():
            if v is None:
                rec["bit_exact"][k] = True
                continue
            got = ttnn.to_torch(run(v))
            rec["bit_exact"][k] = bool(torch.equal(ref, got))
            del got
        del ref

        live = [k for k, v in arms.items() if k == "ttnn" or v is not None]
        for k in live:  # warm every arm before any is timed
            for _ in range(2):
                ttnn.deallocate(run(arms[k]))
        samples = {k: [] for k in live}
        for _ in range(a.reps):
            for k in live:  # round-robin, so drift hits both arms equally
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                for _ in range(a.pipe):
                    ttnn.deallocate(run(arms[k]))
                ttnn.synchronize_device(dev)
                samples[k].append((time.perf_counter() - t0) * 1e3 / a.pipe)
        for k in live:
            rec["ms"][k] = round(med(samples[k]), 4)
            rec.setdefault("ms_all", {})[k] = [round(x, 4) for x in samples[k]]
        base = rec["ms"]["ttnn"]
        rec["speedup"] = {k: round(base / rec["ms"][k], 3) for k in live}
        for k in live:
            print(f"   {k:4s} {rec['ms'][k]:8.4f} ms  {rec['speedup'][k]:5.2f}x  "
                  f"bit-exact={rec['bit_exact'][k]}")

        if a.sweep:
            sw = {}
            for p in range(1, Mt + 1):
                if Mt % p:
                    continue
                if p != Mt and batch * Mt // p > cores:
                    sw[f"per_core_M={p}"] = "unsafe (block-stride rule)"
                    continue
                for bw in sorted({1, 2, Kt}):
                    if Kt % bw:
                        continue
                    try:
                        c = manual_config(p, bw, Nt, grid)
                        r = run(c)
                        ex = bool(torch.equal(ttnn.to_torch(run(None)), ttnn.to_torch(r)))
                        ttnn.deallocate(r)
                        for _ in range(2):
                            ttnn.deallocate(run(c))
                        ttnn.synchronize_device(dev)
                        t0 = time.perf_counter()
                        for _ in range(a.pipe):
                            ttnn.deallocate(run(c))
                        ttnn.synchronize_device(dev)
                        ms = (time.perf_counter() - t0) * 1e3 / a.pipe
                        sw[f"per_core_M={p},in0_block_w={bw}"] = {
                            "ms": round(ms, 4), "bit_exact": ex}
                        print(f"     sweep per_core_M={p} in0_block_w={bw}: "
                              f"{ms:8.4f} ms exact={ex}")
                    except Exception as e:  # a rejected config is a datapoint, not a crash
                        sw[f"per_core_M={p},in0_block_w={bw}"] = f"rejected: {type(e).__name__}"
            rec["sweep"] = sw

        out["shapes"][name] = rec
        ttnn.deallocate(A); ttnn.deallocate(B)

    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

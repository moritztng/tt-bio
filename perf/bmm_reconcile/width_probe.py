#!/usr/bin/env python3
"""E7 — the K-split width for the square non-narrow classes, where ttnn takes its 2D factory.

ttnn 0.67.4 `create_simple_matmul_program_config` picks the K-block width two different ways
(`matmul_program_config.cpp` at tag v0.67.4): the 1D factories take `Kt % 2 == 0 ? 2 : 1`
(`get_mcast_1d_config`, line 330), while the all-DRAM 2D factory starts at
`Kt % num_cores_x == 0 ? Kt / num_cores_x : 1` and then lets `get_multi_dim_per_core_factor`
adjust it (line 1158/1176). The 1D value is computable in Python; the 2D one is not, so the
classes that route 2D have to be measured. `(1,64,298,298)^2` is the one that matters: it is the
only class in the census where the removed `n_tiles > 4` gate would newly admit a config, and the
first sweep found `in0_block_w=2` NOT bit-exact there.
"""
import argparse, json, time
import torch, ttnn
from tt_bio.tenstorrent import get_device
import tt_bio.tenstorrent as T

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32, BF16 = ttnn.float32, ttnn.bfloat16

CASES = [
    ((1, 64, 298, 298), (1, 64, 298, 298), BF16, "trimul class, 2D branch, square"),
    ((1, 16, 320, 320), (1, 16, 320, 64), F32, "DiT attn@v, 2D branch, Nt=2"),
    ((1, 16, 320, 64), (1, 16, 64, 320), F32, "DiT q@kT, 2D branch, Nt=10"),
]


def sub(p, n):
    best = (1, 1)
    for h in range(1, p + 1):
        if p % h:
            continue
        for w in range(1, n + 1):
            if n % w or h * w > 4:
                continue
            if h * w > best[0] * best[1]:
                best = (h, w)
    return best


def med(xs):
    return sorted(xs)[len(xs) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = get_device()
    T._configure_active_compute_grid(dev)
    grid = T.COMPUTE_GRID_MAIN
    cores = grid[0] * grid[1]
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    out = {"grid": list(grid), "cases": []}
    for sa, sb, dt, note in CASES:
        A = ttnn.from_torch(torch.randn(*sa) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=dt, memory_config=DRAM)
        B = ttnn.from_torch(torch.randn(*sb) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=dt, memory_config=DRAM)
        pa, pb = list(A.padded_shape), list(B.padded_shape)
        batch = 1
        for d in pa[:-2]:
            batch *= int(d)
        mt, kt, nt = int(pa[-2]) // 32, int(pa[-1]) // 32, int(pb[-1]) // 32
        H, W = mt * 32, nt * 32
        narrow = (max(H, W) > 8 * min(H, W)) or H <= 32 or W <= 32
        rec = {"a": list(sa), "b": list(sb), "note": note, "batch": batch,
               "Mt": mt, "Kt": kt, "Nt": nt, "narrow": narrow, "arms": {}}
        print(f"\n== {note}  {sa}x{sb} B={batch} Mt/Kt/Nt={mt}/{kt}/{nt} narrow={narrow}")
        ref = ttnn.to_torch(ttnn.matmul(A, B, compute_kernel_config=ckc))
        t0 = time.perf_counter()
        for _ in range(8):
            ttnn.deallocate(ttnn.matmul(A, B, compute_kernel_config=ckc))
        ttnn.synchronize_device(dev)
        base = (time.perf_counter() - t0) * 1e3 / 8
        pipe = max(8, min(400, int(25.0 / max(base, 1e-3))))
        for p in range(1, mt + 1):
            if mt % p or (p != mt and batch * mt // p > cores):
                continue
            for bw in sorted({1, 2, kt} & set(range(1, kt + 1))):
                if kt % bw:
                    continue
                h, w = sub(p, nt)
                cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=grid, in0_block_w=bw,
                    out_subblock_h=h, out_subblock_w=w, per_core_M=p, per_core_N=nt)
                r = ttnn.matmul(A, B, compute_kernel_config=ckc, program_config=cfg)
                ex = bool(torch.equal(ref, ttnn.to_torch(r)))
                ttnn.deallocate(r)
                for _ in range(2):
                    ttnn.deallocate(ttnn.matmul(A, B, compute_kernel_config=ckc,
                                                program_config=cfg))
                s = []
                for _ in range(5):
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    for _ in range(pipe):
                        ttnn.deallocate(ttnn.matmul(A, B, compute_kernel_config=ckc,
                                                    program_config=cfg))
                    ttnn.synchronize_device(dev)
                    s.append((time.perf_counter() - t0) * 1e3 / pipe)
                ms = med(s)
                rec["arms"][f"pM={p},bw={bw}"] = {"ms": round(ms, 5), "bit_exact": ex,
                                                  "blocks": batch * mt // p}
                print(f"   per_core_M={p:2d} in0_block_w={bw:2d}  {ms:9.4f} ms  "
                      f"blocks={batch*mt//p:5d}  exact={ex}")
        # baseline last, same iteration count
        s = []
        for _ in range(5):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(pipe):
                ttnn.deallocate(ttnn.matmul(A, B, compute_kernel_config=ckc))
            ttnn.synchronize_device(dev)
            s.append((time.perf_counter() - t0) * 1e3 / pipe)
        rec["baseline_ms"] = round(med(s), 5)
        rec["iters"] = pipe * 5
        print(f"   ttnn baseline           {rec['baseline_ms']:9.4f} ms  (iters/arm={rec['iters']})")
        out["cases"].append(rec)
        ttnn.deallocate(A); ttnn.deallocate(B)
        del ref
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

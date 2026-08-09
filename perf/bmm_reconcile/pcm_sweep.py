#!/usr/bin/env python3
"""E7 — per_core_M sweep at the bit-exact K-block width, every class the three models issue.

The width probe showed per_core_M does not affect bit-exactness (every legal per_core_M is exact
at the class's own width and wrong at every other width), so per_core_M is a pure performance
knob. G1 picks it by minimising DRAM reads and E1 by maximising occupancy, and the class run
showed those two disagree by up to 1.3x. This sweeps every legal value so the rule is chosen
against the whole curve instead of two points on it.

Legal = G1's block-stride predicate: per_core_M == Mt, or the total block count fits the grid.
"""
import argparse, json, time
import torch, ttnn
from tt_bio.tenstorrent import get_device
import tt_bio.tenstorrent as T

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32, BF16 = ttnn.float32, ttnn.bfloat16

# (a, b, dtype, label, exact in0_block_w measured by classes_c0.py / width_probe.py)
CASES = [
    ((75, 4, 32, 128), (75, 4, 128, 32), F32, "protenix atom q@kT", 2),
    ((75, 4, 32, 32), (75, 4, 32, 128), F32, "protenix atom attn@v", 1),
    ((1, 16, 320, 320), (1, 16, 320, 64), F32, "DiT attn@v fp32", 2),
    ((1, 16, 320, 64), (1, 16, 64, 320), F32, "DiT q@kT fp32", 1),
    ((1, 16, 608, 608), (1, 16, 608, 64), BF16, "opendde DiT attn@v", 1),
    ((1, 8, 608, 608), (1, 8, 608, 64), BF16, "opendde DiT attn@v tail", 1),
    ((298, 4, 298, 298), (298, 4, 298, 32), BF16, "OF3 tri-att attn@v", 2),
    ((298, 4, 298, 32), (298, 4, 32, 298), BF16, "OF3 tri-att q@kT", 1),
    ((1, 16, 298, 298), (1, 16, 298, 32), BF16, "OF3 AttnPairBias attn@v", 2),
    ((1, 75, 4, 32, 32), (1, 75, 4, 32, 128), F32, "OF3 atom attn@v rank5", 1),
    ((1, 75, 4, 32, 128), (1, 75, 4, 128, 32), F32, "OF3 atom q@kT rank5", 2),
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
    l1 = int(ttnn.get_max_worker_l1_unreserved_size())
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    out = {"grid": list(grid), "cores": cores, "l1_unreserved": l1, "cases": []}
    for sa, sb, dt, label, bw in CASES:
        A = ttnn.from_torch(torch.randn(*sa) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=dt, memory_config=DRAM)
        B = ttnn.from_torch(torch.randn(*sb) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=dt, memory_config=DRAM)
        pa, pb = list(A.padded_shape), list(B.padded_shape)
        batch = 1
        for d in pa[:-2]:
            batch *= int(d)
        mt, kt, nt = int(pa[-2]) // 32, int(pa[-1]) // 32, int(pb[-1]) // 32
        eb = 4 if dt == F32 else 2
        tile, acc = 1024 * eb, 4096
        rec = {"a": list(sa), "b": list(sb), "label": label, "dtype": "fp32" if dt == F32 else "bf16",
               "batch": batch, "Mt": mt, "Kt": kt, "Nt": nt, "in0_block_w": bw, "arms": {}}
        print(f"\n== {label}  B={batch} Mt/Kt/Nt={mt}/{kt}/{nt} bw={bw}")
        ref = ttnn.to_torch(ttnn.matmul(A, B, compute_kernel_config=ckc))
        t0 = time.perf_counter()
        for _ in range(8):
            ttnn.deallocate(ttnn.matmul(A, B, compute_kernel_config=ckc))
        ttnn.synchronize_device(dev)
        est = (time.perf_counter() - t0) * 1e3 / 8
        pipe = max(8, min(400, int(25.0 / max(est, 1e-3))))
        for p in range(1, mt + 1):
            if mt % p or (p != mt and batch * mt // p > cores):
                continue
            if 2 * (p + nt) * bw * tile + p * nt * (tile + acc) > l1:
                rec["arms"][f"pM={p}"] = "L1"
                print(f"   per_core_M={p:2d}  does not fit L1")
                continue
            h, w = sub(p, nt)
            cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=grid, in0_block_w=bw,
                out_subblock_h=h, out_subblock_w=w, per_core_M=p, per_core_N=nt)
            r = ttnn.matmul(A, B, compute_kernel_config=ckc, program_config=cfg)
            ex = bool(torch.equal(ref, ttnn.to_torch(r)))
            ttnn.deallocate(r)
            for _ in range(2):
                ttnn.deallocate(ttnn.matmul(A, B, compute_kernel_config=ckc, program_config=cfg))
            s = []
            for _ in range(5):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                for _ in range(pipe):
                    ttnn.deallocate(ttnn.matmul(A, B, compute_kernel_config=ckc,
                                                program_config=cfg))
                ttnn.synchronize_device(dev)
                s.append((time.perf_counter() - t0) * 1e3 / pipe)
            blocks = batch * mt // p
            rec["arms"][f"pM={p}"] = {"ms": round(med(s), 5), "blocks": blocks, "bit_exact": ex}
            print(f"   per_core_M={p:2d}  {med(s):9.4f} ms  blocks={blocks:5d}  exact={ex}")
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
        print(f"   ttnn baseline   {rec['baseline_ms']:9.4f} ms  (iters/arm={rec['iters']})")
        out["cases"].append(rec)
        ttnn.deallocate(A); ttnn.deallocate(B)
        del ref
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

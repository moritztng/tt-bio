#!/usr/bin/env python3
"""Two W2 experiments at the 298 aa shape: the channel-move route, and the output projection.

R (routes). The channel move [1,I,J,C] -> [1,C,I,J] that feeds the triangle matmul runs at
12-16% of this card's L1 copy roof. It is not a whole-tile move: an output tile (i-range,
j-range at one c) draws one 32-element row from each of 32 different input tiles, so the tile
path issues 64-byte transactions. Hypothesis: ROW_MAJOR layout gives the same permutation a
contiguous inner run (J elements = 640 B at N=320) and is therefore faster even after paying
untilize + tilize. Prediction if true: the row-major route beats 0.147 ms. Every route is
checked bit-exact against the production permute.

M (module). ttnn.linear(core_grid=...) gets 20.6 TFLOP/s on [1,320,320,256]@[256,256] where
ttnn.experimental.minimal_matmul gets 35.7 (perf/trimul_kernel/layout_micro.py). The two
trimul output projections are 15.6% of the op. Measure the swap on the real module, not the
micro-benchmark, and report parity against the production output.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:perfwar-trimul-kernel \
        python3 perf/trimul_kernel/w2_routes.py --n 320
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import CORE_GRID_MAIN, get_device  # noqa: E402

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG


def timeit(dev, fn, warm=4, iters=7, pipe=10):
    for _ in range(warm):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    ser = []
    for _ in range(iters):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ser.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = [fn() for _ in range(pipe)]
    ttnn.synchronize_device(dev)
    pip = (time.perf_counter() - t0) * 1e3 / pipe
    for o in outs:
        ttnn.deallocate(o)
    return sorted(ser)[len(ser) // 2], pip


def routes(dev, N, C, out):
    torch.manual_seed(0)
    ch = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=L1)
    nb = N * N * C * 2
    rows = []

    def bench(name, fn, ref=None, nbytes=nb):
        try:
            ser, pip = timeit(dev, fn)
        except Exception as e:
            print(f"{name:52s} FAILED {type(e).__name__}: {str(e)[:110]}")
            rows.append(dict(name=name, error=f"{type(e).__name__}: {str(e)[:220]}"))
            return None
        eq = None
        if ref is not None:
            o = fn()
            eq = bool(torch.equal(ttnn.to_torch(o), ref))
            ttnn.deallocate(o)
        gbs = nbytes / (pip * 1e-3) / 1e9
        rows.append(dict(name=name, serial_ms=round(ser, 4), pipe_ms=round(pip, 4),
                         gbs_each_way=round(gbs, 1), bit_exact=eq))
        print(f"{name:52s} {pip:8.4f} ms {gbs:7.1f} GB/s each way  "
              f"{'exact=' + str(eq) if eq is not None else ''}")
        return pip

    print(f"\nR: channel move [1,{N},{N},{C}] -> [1,{C},{N},{N}], {nb / 2**20:.1f} MiB, L1")
    roof = bench("clone (L1 copy roof)", lambda: ttnn.clone(ch, memory_config=L1))
    ref = ttnn.to_torch(ttnn.permute(ch, (0, 3, 1, 2), memory_config=L1))
    bench("r0 permute(0,3,1,2)  [production]",
          lambda: ttnn.permute(ch, (0, 3, 1, 2), memory_config=L1), ref)

    def r1():
        t = ttnn.transpose(ch, -2, -1, memory_config=L1)
        o = ttnn.permute(t, (0, 2, 1, 3), memory_config=L1)
        ttnn.deallocate(t)
        return o
    bench("r1 transpose(-2,-1) . permute(0,2,1,3)", r1, ref)

    def r2():
        t = ttnn.transpose(ch, -2, -1, memory_config=L1)
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(t)
        p = ttnn.permute(rm, (0, 2, 1, 3))
        ttnn.deallocate(rm)
        o = ttnn.to_layout(p, ttnn.TILE_LAYOUT)
        ttnn.deallocate(p)
        return o
    bench("r2 transpose . [row-major permute(0,2,1,3)] . tilize", r2, ref)

    def r3():
        rm = ttnn.to_layout(ch, ttnn.ROW_MAJOR_LAYOUT)
        p = ttnn.permute(rm, (0, 3, 1, 2))
        ttnn.deallocate(rm)
        o = ttnn.to_layout(p, ttnn.TILE_LAYOUT)
        ttnn.deallocate(p)
        return o
    bench("r3 [row-major permute(0,3,1,2)] . tilize", r3, ref)

    def rt():
        rm = ttnn.to_layout(ch, ttnn.ROW_MAJOR_LAYOUT)
        o = ttnn.to_layout(rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(rm)
        return o
    bench("   price of untilize+tilize alone", rt)
    bench("   untilize alone", lambda: ttnn.to_layout(ch, ttnn.ROW_MAJOR_LAYOUT))
    ttnn.deallocate(ch)
    if roof:
        print(f"   roof = {roof:.4f} ms")
    out.extend(rows)


def module_ab(dev, N, out):
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    layer, c_z = build_layer(ckc)
    torch.manual_seed(0)
    z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    rows = []
    for name in ("tri_mul_start", "tri_mul_end"):
        tm = (layer.triangle_multiplication_start if name == "tri_mul_start"
              else layer.triangle_multiplication_end)
        arms = {}
        for arm, flag in (("linear", False), ("minimal_matmul", True)):
            T._TRIMUL_MM_OUT = flag
            o = tm(z, None)
            h = ttnn.to_torch(o)
            ttnn.deallocate(o)
            ser, pip = timeit(dev, lambda: tm(z, None))
            arms[arm] = (ser, pip, h)
            print(f"{name} {arm:16s} serial {ser:7.3f} ms  pipe {pip:7.3f} ms")
        a, b = arms["linear"][2], arms["minimal_matmul"][2]
        eq = bool(torch.equal(a, b))
        d = (a.float() - b.float())
        rmsd = float(d.pow(2).mean().sqrt())
        pcc = float(torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1])
        print(f"{name} parity: bit_exact={eq} rmsd={rmsd:.5f} (ref std {a.float().std():.4f}) "
              f"PCC={pcc:.7f}")
        rows.append(dict(module=name,
                         linear_serial_ms=round(arms["linear"][0], 3),
                         linear_pipe_ms=round(arms["linear"][1], 3),
                         mm_serial_ms=round(arms["minimal_matmul"][0], 3),
                         mm_pipe_ms=round(arms["minimal_matmul"][1], 3),
                         speedup_pipe=round(arms["linear"][1] / arms["minimal_matmul"][1], 4),
                         bit_exact=eq, rmsd=round(rmsd, 6),
                         ref_std=round(float(a.float().std()), 4), pcc=round(pcc, 7)))
    T._TRIMUL_MM_OUT = True
    out.extend(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--mode", default="all", choices=["all", "routes", "module"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = get_device()
    out = []
    if args.mode in ("all", "routes"):
        routes(dev, args.n, T._trimul_chunk_size(args.n, 128), out)
    if args.mode in ("all", "module"):
        module_ab(dev, args.n, out)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

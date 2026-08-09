#!/usr/bin/env python3
"""The pair-track projections at the shapes production actually builds them at.

perf/pf_matmul/proj_ab.py measured a flattened 102400x256 stand-in. The live fold passes a
batched (1, 298, 298, 256) / (298, 298, 256), so a logical 298 pads to 320 per batch row and
m_tiles is 298x10 = 2980, not 3200. That changes per_core_M, the core count and the byte count,
so the whole ledger has to be rebuilt here. Roofs are measured in this same process on this card.
"""
import argparse, json, time
import torch, ttnn
from tt_bio.tenstorrent import get_device
import tt_bio.tenstorrent as T

DRAM = ttnn.DRAM_MEMORY_CONFIG


def med(xs):
    return sorted(xs)[len(xs) // 2]


def timed(dev, fn, warm=2, pipe=6, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) * 1e3 / pipe)
    return med(o)


def roofs(dev, ckc):
    out = {}
    n = 4096
    a = ttnn.from_torch(torch.randn(1, 1, n, n) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    b = ttnn.from_torch(torch.randn(1, 1, n, n) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    ms = timed(dev, lambda: ttnn.deallocate(ttnn.experimental.minimal_matmul(
        a, b, memory_config=DRAM, compute_kernel_config=ckc)), pipe=4)
    out["compute_square_TFLOPs"] = round(2 * n ** 3 / 1e9 / ms, 2)
    ttnn.deallocate(a); ttnn.deallocate(b)
    nb = 32 * 1024 * 1024
    rows = nb // (2 * 1024)
    t = ttnn.from_torch(torch.zeros(1, 1, rows, 1024), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    ms = timed(dev, lambda: ttnn.deallocate(ttnn.clone(t, memory_config=DRAM)), pipe=4)
    out["dram_rw_clone_GBs"] = round(2 * nb / 1e9 / (ms / 1e3), 1)
    ttnn.deallocate(t)
    # the rate a K=256 contraction can actually reach: best backend on the real class
    return out


# (label, in0 shape, n_out, calls/fold at 298 aa -- counted by infold_pp.py)
def classes(c_z, bias_n):
    S = 298
    return [
        ("trimul.out_proj  p_out+g_out", (1, S, S, c_z), c_z, 2096),
        ("triatt.out       x_out",       (S, S, c_z),    c_z, 1048),
        ("triatt.bias      heads",       (S, S, c_z),    bias_n, 1048),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-z", type=int, default=256)
    ap.add_argument("--bias-n", type=int, default=8)
    ap.add_argument("--bias-pad", type=int, default=0,
                    help="widen the bias output to this many columns (step 5: is nt=1 the limiter?)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    res = dict(card="qb1 card 1", grid=list(T.COMPUTE_GRID_MAIN), roofs=roofs(dev, ckc), rows=[])
    nc = T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]

    for label, xs, n_out, calls in classes(a.c_z, a.bias_n):
        x = ttnn.from_torch(torch.randn(*xs) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        w = ttnn.from_torch(torch.randn(xs[-1], n_out) * 0.1, layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
        batch = 1
        for d in xs[:-2]:
            batch *= d
        mt = batch * -(-xs[-2] // 32)
        kt = -(-xs[-1] // 32)
        nt = -(-n_out // 32)
        flops = 2 * (mt * 32) * (kt * 32) * (nt * 32)
        byts = (mt * kt + kt * nt + mt * nt) * 1024 * 2

        def run(cfg):
            if cfg is None:
                return lambda: ttnn.deallocate(ttnn.linear(
                    x, w, memory_config=DRAM, dtype=ttnn.bfloat16,
                    compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN))
            return lambda: ttnn.deallocate(ttnn.linear(
                x, w, memory_config=DRAM, dtype=ttnn.bfloat16,
                compute_kernel_config=ckc, program_config=cfg))

        arms = {}
        arms["prod_core_grid"] = (None, timed(dev, run(None)))
        for bw in (1, 2, 4, 8, kt):
            if kt % bw or bw > kt:
                continue
            cfg = T._pair_proj_program_config(mt, kt, nt, bw, 2)
            if cfg is None:
                continue
            arms[f"cfg_bw{bw}"] = (cfg, timed(dev, run(cfg)))
        # minimal_matmul for reference
        try:
            arms["minimal_matmul"] = (None, timed(dev, lambda: ttnn.deallocate(
                ttnn.experimental.minimal_matmul(x, w, memory_config=DRAM,
                                                 dtype=ttnn.bfloat16,
                                                 compute_kernel_config=ckc))))
        except Exception as e:
            arms["minimal_matmul"] = (None, None)

        # bit-exactness of every cfg arm against production, on these operands
        prod = ttnn.linear(x, w, memory_config=DRAM, dtype=ttnn.bfloat16,
                           compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
        pt = ttnn.to_torch(prod)
        exact = {}
        for k, (cfg, _) in arms.items():
            if cfg is None:
                continue
            g = ttnn.linear(x, w, memory_config=DRAM, dtype=ttnn.bfloat16,
                            compute_kernel_config=ckc, program_config=cfg)
            exact[k] = bool(torch.equal(pt, ttnn.to_torch(g)))
            ttnn.deallocate(g)
        ttnn.deallocate(prod)

        cfg1 = T._pair_proj_program_config(mt, kt, nt, 1, 2)
        row = dict(
            label=label, in0=list(xs), n_out=n_out, mt=mt, kt=kt, nt=nt,
            per_core_M=cfg1.per_core_M if cfg1 else None,
            blocks=-(-mt // cfg1.per_core_M) if cfg1 else None, num_cores=nc,
            GFLOP=round(flops / 1e9, 3), MB=round(byts / 1e6, 2),
            AI_flop_per_byte=round(flops / byts, 1), calls_per_fold=calls,
            ms={k: (round(v, 4) if v else None) for k, (c, v) in arms.items()},
            bit_exact=exact,
        )
        for k, (c, v) in arms.items():
            if v:
                row.setdefault("TFLOPs", {})[k] = round(flops / 1e9 / v, 2)
                row.setdefault("GBs", {})[k] = round(byts / 1e6 / v, 1)
        res["rows"].append(row)
        print(json.dumps(row, indent=2), flush=True)
        ttnn.deallocate(x); ttnn.deallocate(w)

    if a.bias_pad:
        S = 298
        x = ttnn.from_torch(torch.randn(S, S, a.c_z) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        out = []
        for n_out in (a.bias_n, 32, 64, 128):
            w = ttnn.from_torch(torch.randn(a.c_z, n_out) * 0.1, layout=ttnn.TILE_LAYOUT,
                                device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
            mt, kt, nt = 298 * 10, a.c_z // 32, -(-n_out // 32)
            cfg = T._pair_proj_program_config(mt, kt, nt, 1, 2)
            ms = timed(dev, lambda: ttnn.deallocate(ttnn.linear(
                x, w, memory_config=DRAM, dtype=ttnn.bfloat16,
                compute_kernel_config=ckc, program_config=cfg)))
            byts = (mt * kt + kt * nt + mt * nt) * 1024 * 2
            out.append(dict(n_out=n_out, nt=nt, ms=round(ms, 4), MB=round(byts / 1e6, 2),
                            GBs=round(byts / 1e6 / ms, 1)))
            print(out[-1], flush=True)
            ttnn.deallocate(w)
        res["bias_width_sweep"] = out
        ttnn.deallocate(x)

    open(a.out, "w").write(json.dumps(res, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

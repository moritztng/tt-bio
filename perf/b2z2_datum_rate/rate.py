"""Measure the Tensix datum rate directly: ns per tile for an L1 -> DST -> L1 move, nothing else.

Wave 1 priced Boltz-2's trunk at 71.3 ns/tile against a *derived* 64 ns/tile floor (1024 datums per
tile / ~16 datums per cycle at an assumed 1 GHz). Neither the 16/cycle nor the clock was ever checked
against the part. This runs a kernel that does nothing but move tiles and reads the rate off a slope.

The rate is a SLOPE over the repetition count, not a single timing divided by a tile count: two calls
at different `reps` differ by exactly the loop, so compile, dispatch, program setup and the one-off
DRAM fill cancel instead of being assumed small.

Usage:
  TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:b2z2-datum-rate-floor \
    python3 perf/b2z2_datum_rate/rate.py --out perf/b2z2_datum_rate/results/bh_card2.json
"""

import argparse
import json
import os
import time

import ttnn

KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")

IN_CB = 0
IN2_CB = 1
OUT_CB = 16

DTYPES = {
    "bf16": (ttnn.bfloat16, 32 * 32 * 2),
    "fp32": (ttnn.float32, 32 * 32 * 4),
    "bfp8_b": (ttnn.bfloat8_b, 32 * 32 + 64),
}

FIDELITY = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}


def build(device, x, out, cfg):
    nt, gran, out_slots, mode = cfg["nt"], cfg["gran"], cfg["out_slots"], cfg["mode"]
    dt, tile_bytes = DTYPES[cfg["dtype"]]
    gx, gy = cfg["grid"]
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]
    )

    def cb(idx, depth):
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=tile_bytes)
        return ttnn.CBDescriptor(
            total_size=depth * tile_bytes, core_ranges=core_grid, format_descriptors=[fmt]
        )

    cbs = [cb(IN_CB, nt), cb(OUT_CB, out_slots)]
    if mode == 4:
        cbs.append(cb(IN2_CB, nt))

    reader_ct = [IN_CB, nt, IN2_CB, 1 if mode == 4 else 0] + list(
        ttnn.TensorAccessorArgs(x).get_compile_time_args()
    )
    writer_ct = [OUT_CB, out_slots] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for cx in range(gx):
        for cy in range(gy):
            reader_rt[cx][cy] = [x.buffer_address()]
            compute_rt[cx][cy] = [cfg["reps"]]
            writer_rt[cx][cy] = [out.buffer_address(), 1 if (cx == 0 and cy == 0) else 0]

    reader = ttnn.KernelDescriptor(
        kernel_source=os.path.join(KERNEL_DIR, "reader_fill.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=reader_ct, runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=os.path.join(KERNEL_DIR, "writer_dump.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=writer_ct, runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=os.path.join(KERNEL_DIR, "compute_move.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[IN_CB, OUT_CB, gran, nt, out_slots, mode, IN2_CB],
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=FIDELITY[cfg["fidelity"]], fp32_dest_acc_en=cfg["fp32_dest"]
        ),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)


def run_once(device, cfg, reps, reps_warm=True):
    """One generic_op call at a given rep count. Returns (seconds, output tensor)."""
    dt, _ = DTYPES[cfg["dtype"]]
    nt, out_slots = cfg["nt"], cfg["out_slots"]
    x = ttnn.from_torch(
        _torch_src(nt), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(
        _torch_zero(out_slots), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    c = dict(cfg)
    c["reps"] = reps
    pd = build(device, x, out, c)
    if reps_warm:
        ttnn.generic_op([x, out], pd)
        ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    ttnn.generic_op([x, out], pd)
    ttnn.synchronize_device(device)
    dt_s = time.perf_counter() - t0
    return dt_s, out


def _torch_src(nt):
    import torch

    g = torch.Generator().manual_seed(0)
    return torch.randn(1, 1, 32, 32 * nt, generator=g)


def _torch_zero(n):
    import torch

    return torch.zeros(1, 1, 32, 32 * n)


def measure(device, cfg, reps_lo, reps_hi, n=3):
    """ns/tile from the slope between two rep counts, taking the min of n samples at each."""
    import torch

    lo = min(run_once(device, cfg, reps_lo)[0] for _ in range(n))
    hi_t, out = None, None
    for _ in range(n):
        t, o = run_once(device, cfg, reps_hi)
        hi_t = t if hi_t is None else min(hi_t, t)
        out = o
    cores = cfg["grid"][0] * cfg["grid"][1]
    tiles_delta = cores * cfg["nt"] * (reps_hi - reps_lo)
    ns_per_tile = (hi_t - lo) * 1e9 / tiles_delta
    packed = bool(ttnn.to_torch(out).abs().sum().item() > 0)
    return {
        "ns_per_tile": ns_per_tile,
        "t_lo_s": lo,
        "t_hi_s": hi_t,
        "tiles_delta": tiles_delta,
        "packed_nonzero": packed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps-lo", type=int, default=4000)
    ap.add_argument("--reps-hi", type=int, default=20000)
    ap.add_argument("--nt", type=int, default=64)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--arms", default="core")
    args = ap.parse_args()

    from tt_bio import tenstorrent as TT

    device = TT.get_device()
    results = []
    try:
        base = dict(nt=args.nt, out_slots=8, gran=8, mode=0, dtype="bf16",
                    fidelity="LoFi", fp32_dest=False, grid=(1, 1), reps=0)
        arms = []
        for mode in (3, 4):
            arms.append(("mode%d" % mode, dict(base, mode=mode)))
        for nt in (8, 32, 128):
            arms.append(("nt%d" % nt, dict(base, nt=nt)))
        arms.append(("binary_gran1", dict(base, mode=4, gran=1)))
        arms.append(("binary_grid_full", dict(base, mode=4, grid=(11, 10))))
        arms.append(("binary_fp32dest", dict(base, mode=4, gran=4, fp32_dest=True)))
        arms.append(("binary_hifi4", dict(base, mode=4, fidelity="HiFi4")))
        arms.append(("binary_bfp8", dict(base, mode=4, dtype="bfp8_b")))
        for mode in (0, 1, 2):
            arms.append(("mode%d" % mode, dict(base, mode=mode)))
        for gran in (1, 2, 4, 8):
            arms.append(("gran%d" % gran, dict(base, gran=gran)))
        for dtype in ("bf16", "fp32", "bfp8_b"):
            arms.append(("dtype_%s" % dtype, dict(base, dtype=dtype)))
            arms.append(("dtype_%s_packonly" % dtype, dict(base, dtype=dtype, mode=2)))
        arms.append(("fp32dest", dict(base, fp32_dest=True, gran=4)))
        arms.append(("hifi4", dict(base, fidelity="HiFi4")))
        arms.append(("grid_full", dict(base, grid=(11, 10))))

        for name, cfg in arms:
            reps_lo, reps_hi = args.reps_lo, args.reps_hi
            cores = cfg["grid"][0] * cfg["grid"][1]
            if cores > 1:
                reps_lo, reps_hi = reps_lo // 16, reps_hi // 16
            t0 = time.perf_counter()
            try:
                r = measure(device, cfg, reps_lo, reps_hi, n=args.n)
            except Exception as e:  # an arm that will not compile must not lose the rest
                r = {"error": repr(e)[:300]}
            r.update(name=name, cfg={k: v for k, v in cfg.items() if k != "reps"},
                     reps=(reps_lo, reps_hi), wall_s=time.perf_counter() - t0)
            results.append(r)
            print(json.dumps(r), flush=True)
    finally:
        try:
            TT.cleanup()
        except Exception:
            pass
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()

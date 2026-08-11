#!/usr/bin/env python3
"""Roofs on this card, then the three trimul components at 512 aa, measured.

Nothing here is inherited: every roof is re-measured in this process, and every component is
timed at the shape and buffer type production actually gives it at 512 aa (checked, not assumed).
Byte figures are DERIVED from shapes + the verified buffer_type of each operand; the installed
ttnn wheel has no device profiler (DumpDeviceProfileResults is absent), so bytes-at-the-controller
are not available. Every derived byte figure is divided by its measured time and checked against
the roofs measured here.

    TT_VISIBLE_DEVICES=0 python3 perf/trimul_root/probe.py --n 512 --out perf/trimul_root/x.json
"""
import argparse, json, statistics as st, time
from pathlib import Path

import torch
import ttnn

import tt_bio.tenstorrent as T
from tt_bio.tenstorrent import CORE_GRID_MAIN, COMPUTE_GRID_MAIN, get_device


def median_pipe(dev, fn, warm=3, iters=5, pipe=6):
    """Serial median (sync both sides) and pipelined mean (no per-call sync)."""
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
    return st.median(ser), pip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    N = args.n
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    gx, gy = COMPUTE_GRID_MAIN
    res = {"n": N, "grid": [gx, gy], "cores": gx * gy,
           "l1_bank_bytes": T._l1_bank_bytes(),
           "roofs": {}, "components": {}, "contraction_configs": []}
    print(f"grid {gx}x{gy} = {gx * gy} cores", flush=True)

    # ---------------------------------------------------------------- roofs
    def clone_roof(src_mem, dst_mem, mb):
        rows = int(mb * 1e6 / 2) // 4096
        nb = rows * 4096 * 2
        x = ttnn.from_torch(torch.randn(rows, 4096), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=src_mem)
        _, pip = median_pipe(dev, lambda: ttnn.clone(x, memory_config=dst_mem))
        ttnn.deallocate(x)
        return nb, pip

    for name, s, d, mb in [("dram_read_gbs", ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, 48),
                           ("dram_write_gbs", ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, 32),
                           ("dram_rw_gbs", ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, 48),
                           ("l1_copy_gbs_each_way", ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, 32)]:
        try:
            nb, ms = clone_roof(s, d, mb)
            v = nb / (ms * 1e-3) / 1e9
            if name == "dram_rw_gbs":
                v *= 2
            res["roofs"][name] = round(v, 1)
            res["roofs"][name + "_mb"] = round(nb / 1e6, 1)
            print(f"roof {name}: {v:.1f} GB/s ({nb/1e6:.1f} MB, {ms:.3f} ms)", flush=True)
        except Exception as e:
            res["roofs"][name + "_err"] = str(e)[:160]
            print(f"roof {name}: ERR {e}"[:200], flush=True)

    # square compute roof, DRAM output
    for M in (2048, 4096):
        try:
            a = ttnn.from_torch(torch.randn(M, M), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            b = ttnn.from_torch(torch.randn(M, M), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _, pip = median_pipe(dev, lambda: ttnn.matmul(
                a, b, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16))
            tf = 2 * M ** 3 / (pip * 1e-3) / 1e12
            res["roofs"][f"square_compute_tflops_{M}"] = round(tf, 2)
            print(f"roof square compute {M}^3: {tf:.2f} TFLOP/s ({pip:.3f} ms)", flush=True)
            ttnn.deallocate(a); ttnn.deallocate(b)
        except Exception as e:
            res["roofs"][f"square_compute_{M}_err"] = str(e)[:160]

    # ------------------------------------------------- the three components
    mc = T._triangle_mul_memory_config(N)
    C = T._trimul_chunk_size(N, 128)
    res["buffer_type"] = str(mc.buffer_type)
    res["chunk"] = C
    print(f"production at N={N}: buffer={mc.buffer_type} chunk={C}", flush=True)
    c_z = 256
    pair_b = N * N * c_z * 2
    chunk_b = N * N * C * 2

    z = ttnn.from_torch(torch.randn(1, N, N, c_z), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # in_proj: the fused [g_a|g_b|p_a|p_b] input projection, c_z -> 4C
    w_in = ttnn.from_torch(torch.randn(c_z, 4 * C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    _, pip = median_pipe(dev, lambda: ttnn.experimental.minimal_matmul(
        z, w_in, memory_config=mc, dtype=ttnn.bfloat16, compute_kernel_config=ckc))
    fl = 2 * N * N * c_z * 4 * C
    res["components"]["in_proj"] = dict(
        shape=f"1x{N}x{N}x{c_z}@{c_z}x{4*C}", ms=round(pip, 4),
        tflops=round(fl / (pip * 1e-3) / 1e12, 2), flop=fl,
        rbytes=pair_b + c_z * 4 * C * 2, wbytes=4 * chunk_b,
        read_gbs=round((pair_b) / (pip * 1e-3) / 1e9, 1),
        write_gbs=round(4 * chunk_b / (pip * 1e-3) / 1e9, 1))
    print("in_proj: " + json.dumps(res["components"]["in_proj"]), flush=True)

    # out_proj: the pair-track output projection, c_z -> c_z, DRAM out
    w_out = ttnn.from_torch(torch.randn(c_z, c_z), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    _, pip = median_pipe(dev, lambda: ttnn.linear(
        z, w_out, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
        compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN))
    fl = 2 * N * N * c_z * c_z
    res["components"]["out_proj"] = dict(
        shape=f"1x{N}x{N}x{c_z}@{c_z}x{c_z}", ms=round(pip, 4),
        tflops=round(fl / (pip * 1e-3) / 1e12, 2), flop=fl,
        rbytes=pair_b, wbytes=pair_b,
        read_gbs=round(pair_b / (pip * 1e-3) / 1e9, 1),
        write_gbs=round(pair_b / (pip * 1e-3) / 1e9, 1))
    print("out_proj: " + json.dumps(res["components"]["out_proj"]), flush=True)
    ttnn.deallocate(z); ttnn.deallocate(w_in); ttnn.deallocate(w_out)

    # tri_matmul: the contraction, at the production shape and buffer type
    Kt = (N + 31) // 32
    prod_cfg = T._triangle_mul_program_config(Kt)
    print(f"production contraction config: in0_block_w={prod_cfg.in0_block_w} "
          f"per_core_M={prod_cfg.per_core_M} per_core_N={prod_cfg.per_core_N} "
          f"grid={prod_cfg.compute_with_storage_grid_size}", flush=True)
    fl = 2 * C * N * N * N
    a = ttnn.from_torch(torch.randn(1, C, N, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=mc)
    b = ttnn.from_torch(torch.randn(1, C, N, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=mc)

    def cfg(pm, pn, bw, g):
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=g, in0_block_w=bw, out_subblock_h=1, out_subblock_w=1,
            out_block_h=pm, out_block_w=pn, per_core_M=pm, per_core_N=pn,
            transpose_mcast=False, fused_activation=None, fuse_batch=False)

    trials = [("production", prod_cfg, None),
              ("pcM2N2_grid8x8", cfg(2, 2, 8, (8, 8)), None),
              ("pcM2N2_grid10x10", cfg(2, 2, 8, (10, 10)), None),
              ("pcM1N1", cfg(1, 1, 8, (gx, gy)), None),
              ("pcM1N2", cfg(1, 2, 8, (gx, gy)), None),
              ("pcM2N1", cfg(2, 1, 8, (gx, gy)), None),
              ("pcM2N2_bw16", cfg(2, 2, 16, (gx, gy)), None),
              ("pcM1N1_bw16", cfg(1, 1, 16, (gx, gy)), None),
              ("pcM1N1_bw1", cfg(1, 1, 1, (gx, gy)), None),
              ("no_config", None, None),
              ("core_grid", None, CORE_GRID_MAIN)]
    for name, pc, cg in trials:
        try:
            kw = dict(compute_kernel_config=ckc, memory_config=mc, dtype=ttnn.bfloat16)
            if pc is not None:
                kw["program_config"] = pc
            if cg is not None:
                kw["core_grid"] = cg
            ser, pip = median_pipe(dev, lambda: ttnn.matmul(a, b, **kw))
            row = dict(name=name, ms=round(pip, 4), serial_ms=round(ser, 4),
                       tflops=round(fl / (pip * 1e-3) / 1e12, 2),
                       read_gbs=round(2 * chunk_b / (pip * 1e-3) / 1e9, 1),
                       write_gbs=round(chunk_b / (pip * 1e-3) / 1e9, 1))
            if pc is not None:
                row["cfg"] = dict(bw=pc.in0_block_w, pm=pc.per_core_M, pn=pc.per_core_N,
                                  grid=str(pc.compute_with_storage_grid_size))
                row["blocks"] = (-(-Kt // pc.per_core_M)) * (-(-Kt // pc.per_core_N))
            res["contraction_configs"].append(row)
            print("contraction " + json.dumps(row), flush=True)
        except Exception as e:
            res["contraction_configs"].append(dict(name=name, err=str(e)[:200]))
            print(f"contraction {name}: ERR {str(e)[:180]}", flush=True)

    # same contraction with an L1 output, to price the buffer-type term at this size
    try:
        ser, pip = median_pipe(dev, lambda: ttnn.matmul(
            a, b, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16, program_config=prod_cfg))
        res["contraction_configs"].append(dict(
            name="production_L1_out", ms=round(pip, 4),
            tflops=round(fl / (pip * 1e-3) / 1e12, 2)))
        print(f"contraction production_L1_out: {pip:.4f} ms "
              f"{fl / (pip * 1e-3) / 1e12:.2f} TFLOP/s", flush=True)
    except Exception as e:
        res["contraction_configs"].append(dict(name="production_L1_out", err=str(e)[:200]))
        print(f"contraction production_L1_out: ERR {str(e)[:180]}", flush=True)

    ttnn.deallocate(a); ttnn.deallocate(b)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
        print("wrote " + args.out, flush=True)
    print("RESULT_JSON " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()

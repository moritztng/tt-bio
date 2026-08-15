#!/usr/bin/env python3
"""E8i -- is unpack_to_dest per-CB, so an FPU op and an SFPU op can live in one kernel?

E8e/E8f measured every fp32 tile op at 6.99e-4 or worse, including a copy_tile pass-through that does
no arithmetic at all, which places the loss in the unpack path rather than in any adder or
multiplier: the operand is truncated to about 11 mantissa bits on the way into SrcA. None of those
probes set ttnn.ComputeConfigDescriptor.unpack_to_dest_mode, the knob that brings fp32 to DST without
going through SrcA. This is the arm that decides whether §6's 1e-5 gate is reachable at all, and if so
on which unit.

Two kernels, both swept with the knob off and on, at HiFi4 with fp32 dest-acc:
  compute_e8_prec.cpp      FPU   mul_tiles, add_tiles, matmul_tiles -- these read SrcA/SrcB by
                                 construction, so the knob should NOT help them
  compute_e8_sfpuprec.cpp  SFPU  mul_binary_tile, add_binary_tile, and the copy_tile pass-through
                                 that is the instrument -- it has to go to zero

The pass-through is the load-bearing row. If it goes to zero the loss was configuration and the SFPU
numbers below it become real measurements of the SFPU; if it does not, it is the silicon and the
exact-trilinear interpolant has no home on this hardware at a 1e-5 residual.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels"
IN_CB, OUT_CB, TB = 0, 16, 4096


def run(dev, x, out, kern, op, u2d):
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def cb(i):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=TB)
        return ttnn.CBDescriptor(total_size=2 * TB, core_ranges=cg, format_descriptors=[f])

    sa = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    da = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rrt[0][0] = [x.buffer_address(), 0]
    crt[0][0] = [0]
    wrt[0][0] = [out.buffer_address(), 0]
    mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    cc.fp32_dest_acc_en = True
    if u2d:
        m = [ttnn.UnpackToDestMode.Default] * 64; m[1] = ttnn.UnpackToDestMode.UnpackToDestFp32; cc.unpack_to_dest_mode = m
    pd = ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_e8_fill.cpp", [IN_CB, TB] + sa, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / kern, [IN_CB, OUT_CB, op], crt, cc),
        mk(KDIR / "writer_e8_drain.cpp", [OUT_CB, TB] + da, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=[cb(IN_CB), cb(OUT_CB)])
    ttnn.generic_op([x, out], pd)
    ttnn.synchronize_device(dev)
    return ttnn.to_torch(out)[0, 0]


def main():
    dev = ttnn.open_device(device_id=0)
    res = {}
    try:
        t = torch.arange(1024, dtype=torch.float32).reshape(1, 1, 32, 32) * 0.01 + 0.5
        a = t[0, 0]
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        cases = (
            ("compute_e8_prec.cpp", 0, "FPU  mul_tiles", a * a),
            ("compute_e8_prec.cpp", 1, "FPU  add_tiles", a + a),
            ("compute_e8_prec.cpp", 2, "FPU  matmul_tiles", a @ a),
            ("compute_e8_sfpuprec.cpp", 2, "SFPU copy_tile passthrough", a),
            ("compute_e8_sfpuprec.cpp", 0, "SFPU mul_binary_tile", a * a),
            ("compute_e8_sfpuprec.cpp", 1, "SFPU add_binary_tile", a + a),
        )
        for kern, op, name, exact in cases:
            scale = exact.abs().max().item()
            row = {}
            for u2d in (False, True):
                x = ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                    device=dev, memory_config=mc)
                out = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.float32,
                                      layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
                got = run(dev, x, out, kern, op, u2d)
                row["u2d_on" if u2d else "u2d_off"] = (got - exact).abs().max().item() / scale
                ttnn.deallocate(x)
                ttnn.deallocate(out)
            res[name] = row
            print("%-30s off %.3e   on %.3e" % (name, row["u2d_off"], row["u2d_on"]), flush=True)
    finally:
        ttnn.close_device(dev)
    (HERE / "e8i_percb.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

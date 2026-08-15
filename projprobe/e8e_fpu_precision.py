#!/usr/bin/env python3
"""E8e -- how accurate is one fp32 FPU tile op on this silicon, per fidelity?

E8d ran two mul_tiles into one DST slot to settle accumulate-versus-overwrite and, incidentally,
came out 2e-3 to 4e-3 away from the fp32 answer. That is bfloat16 territory (2^-8 = 3.9e-3), not
float32, and §6's gate is a 1e-5 relative residual. So the number has to be nailed down before it
reaches the kernel: either the fidelity knob was not doing what the design assumes, or an fp32
eltwise multiply on this hardware simply is not fp32-accurate, and in that case the design's whole
"fp32 costs 1.004x bf16 so there is nothing to trade" conclusion changes meaning.

One op, not two, so accumulation is not a confound. Swept over the four fidelities. Graded against
torch fp32 and, as the reference point that makes the number readable, against the same computation
with both inputs rounded to bfloat16.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels"
IN_CB, OUT_CB, TB = 0, 16, 4096

FIDELITIES = (("LoFi", ttnn.MathFidelity.LoFi), ("HiFi2", ttnn.MathFidelity.HiFi2),
              ("HiFi3", ttnn.MathFidelity.HiFi3), ("HiFi4", ttnn.MathFidelity.HiFi4))


def run(dev, x, out, op, fid, fp32acc):
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
    cc.math_fidelity = fid
    cc.fp32_dest_acc_en = fp32acc
    pd = ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_e8_fill.cpp", [IN_CB, TB] + sa, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_e8_prec.cpp", [IN_CB, OUT_CB, op], crt, cc),
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
        ab = a.to(torch.bfloat16).to(torch.float32)
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        cases = ((0, "mul_tiles", a * a, ab * ab),
                 (1, "add_tiles", a + a, ab + ab),
                 (2, "matmul_tiles", a @ a, ab @ ab))
        for op, name, exact, bf in cases:
            scale = exact.abs().max().item()
            for fname, fid in FIDELITIES:
                for fp32acc in (True, False):
                    x = ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                        device=dev, memory_config=mc)
                    out = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.float32,
                                          layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
                    got = run(dev, x, out, op, fid, fp32acc)
                    r_fp32 = (got - exact).abs().max().item() / scale
                    r_bf16 = (bf - exact).abs().max().item() / scale
                    key = "%s/%s/acc%d" % (name, fname, int(fp32acc))
                    res[key] = {"rel_vs_fp32": r_fp32, "bf16_would_be": r_bf16}
                    print("%-30s rel %.3e   (bf16 reference %.3e)" % (key, r_fp32, r_bf16),
                          flush=True)
                    ttnn.deallocate(x)
                    ttnn.deallocate(out)
    finally:
        ttnn.close_device(dev)
    (HERE / "e8e_fpu_precision.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

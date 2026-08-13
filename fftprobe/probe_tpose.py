#!/usr/bin/env python3
"""Is transpose_wh_dest bit-exact in fp32? The precision obstacle named on tt-metal #21412."""
import json, time
from pathlib import Path
import torch, ttnn

KDIR = Path(__file__).resolve().parent / "s1b_kernels"
IN_CB, OUT_CB = 0, 16
TB = 32 * 32 * 4


def run(dev, x, out, is_32bit, use_fpu):
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def cb(i, d):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=TB)
        return ttnn.CBDescriptor(total_size=d * TB, core_ranges=cg, format_descriptors=[f])

    rct = [IN_CB, TB, 2] + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    wct = [OUT_CB, TB] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rrt[0][0] = [x.buffer_address(), 0]
    crt[0][0] = [0]
    wrt[0][0] = [out.buffer_address(), 0]
    mk = lambda s, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(KDIR / s), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    pd = ttnn.ProgramDescriptor(kernels=[
        mk("reader_s1b.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk("writer_s1b.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "compute_tpose.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cg, compile_time_args=[IN_CB, OUT_CB, is_32bit, use_fpu], runtime_args=crt,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                                fp32_dest_acc_en=True)),
    ], semaphores=[], cbs=[cb(IN_CB, 2), cb(OUT_CB, 2)])
    ttnn.generic_op([x, out], pd)
    ttnn.synchronize_device(dev)
    return ttnn.to_torch(out).clone()


def main():
    dev = ttnn.open_device(device_id=0)
    res = {}
    try:
        torch.manual_seed(0)
        # Full-mantissa random values: a value that happens to fit 11 bits cannot reveal truncation.
        xt = torch.randn(1, 1, 64, 32, dtype=torch.float32)
        x = ttnn.from_torch(xt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        out = ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.float32),
                              dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        ref = xt[0, 0, :32, :].T.contiguous()
        for name, (b32, fpu) in {"dest_32bit": (1, 0), "dest_16bit": (0, 0),
                                 "fpu_transpose_wh": (0, 1)}.items():
            try:
                got = run(dev, x, out, b32, fpu)[0, 0]
                d = (got - ref)
                res[name] = {
                    "bit_exact": bool(torch.equal(got, ref)),
                    "rel_l2": float(d.norm() / ref.norm()),
                    "max_abs": float(d.abs().max()),
                    "n_differing": int((d != 0).sum()),
                }
                print(f"{name:18s} exact={res[name]['bit_exact']}  rel_l2="
                      f"{res[name]['rel_l2']:.3e}  differing={res[name]['n_differing']}/1024",
                      flush=True)
            except Exception as e:                                       # noqa: BLE001
                res[name] = {"error": str(e)[:300]}
                print(f"{name:18s} ERROR {str(e)[:150]}", flush=True)
        # the wheel's own op, for the same input, as the published baseline
        try:
            t = ttnn.to_torch(ttnn.transpose(x, -2, -1))[0, 0, :32, :32]
            r2 = xt[0, 0, :64, :].T.contiguous()[:32, :32]
            res["ttnn.transpose"] = {"rel_l2": float((t - r2).norm() / r2.norm())}
            print(f"ttnn.transpose     rel_l2={res['ttnn.transpose']['rel_l2']:.3e}", flush=True)
        except Exception as e:                                           # noqa: BLE001
            res["ttnn.transpose"] = {"error": str(e)[:200]}
        json.dump(res, open(Path(__file__).resolve().parent / "probe_tpose.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)


main()

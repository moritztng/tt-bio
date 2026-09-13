"""Does the deferral survive --fast, where the matmul's operands are bfloat8_b?

The split path is the only channel-move branch `--fast` can take (both gated branches require
`not _FAST_MODE`), so widening the deferral to it puts the flag on block-float operands for the
first time. bfloat8_b is a tiled format with a shared exponent per face, so an in-kernel operand
transpose has to re-encode rather than just re-address, and ttnn may refuse or may quietly differ.
Checked at both geometries the split path actually sees: the 298 aa L1 chunk and the 512 aa DRAM
chunk --fast leaves in DRAM.
"""
import json
import statistics as st
import time

import torch

torch.set_grad_enabled(False)
import ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
ckc = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
res = {}


def timed(fn, reps=5):
    ts, out = [], None
    for i in range(reps + 1):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        if i:
            ts.append((time.perf_counter() - t0) * 1e3)
        if out is None:
            out = o
        else:
            ttnn.deallocate(o)
    return st.median(ts), out


def case(tag, S, C, mc, dt):
    pc = T._triangle_mul_program_config(S // 32)
    ah = torch.randn(1, C, S, S, dtype=torch.float32).bfloat16()
    bh = torch.randn(1, C, S, S, dtype=torch.float32).bfloat16()
    a = ttnn.from_torch(ah, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt, memory_config=mc)
    b = ttnn.from_torch(bh, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt, memory_config=mc)

    def mm(x, y, **kw):
        return ttnn.matmul(x, y, compute_kernel_config=ckc, memory_config=mc,
                           program_config=pc, dtype=ttnn.bfloat16, **kw)

    def today():
        bt = ttnn.transpose(b, -2, -1, memory_config=mc)
        o = mm(a, bt)
        ttnn.deallocate(bt)
        return o

    ms_t, ref = timed(today)
    row = {"today_ms": ms_t, "dtype": str(dt), "shape": [1, C, S, S],
           "space": str(mc.buffer_type).split(".")[-1]}
    try:
        ms_d, out = timed(lambda: mm(a, b, transpose_b=True))
        x = ttnn.to_torch(out).float()
        r = ttnn.to_torch(ref).float()
        row.update(defer_ms=ms_d, speedup=ms_t / ms_d, equal=bool(torch.equal(x, r)),
                   max_abs=(x - r).abs().max().item())
        print("%-28s today %8.4f  defer %8.4f  %.4fx  equal=%s max_abs=%.6g" % (
            tag, ms_t, ms_d, ms_t / ms_d, row["equal"], row["max_abs"]), flush=True)
        ttnn.deallocate(out)
    except Exception as e:                                                      # noqa: BLE001
        row["refused"] = str(e)[:400]
        print("%-28s REFUSED: %s" % (tag, str(e)[:250]), flush=True)
    ttnn.deallocate(ref)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    res[tag] = row


case("bf8_b 320 L1", 320, 32, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat8_b)
case("bf8_b 512 DRAM", 512, 128, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat8_b)
json.dump(res, open("/home/ttuser/scratch/uod/mm_transpose_bf8.json", "w"), indent=1)
print("WROTE mm_transpose_bf8.json", flush=True)

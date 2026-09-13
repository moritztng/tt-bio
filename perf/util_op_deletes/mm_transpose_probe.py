"""Can the trimul's matmul take its operand transposed instead of paying a separate transpose?

`_transform_chunk_gated` runs the fused channel move, which produces permute (0,3,1,2), and then a
separate `ttnn.transpose(-2,-1)` whenever the caller wanted (0,3,2,1). At 512 aa that transpose is
a 67.1 MB DRAM -> DRAM copy and the ledger confirms it is a pure relayout: same shape, same space,
one call per trimul, two per block.

The only consumer of the transposed chunk is `ttnn.matmul`, and `ttnn.matmul` takes `transpose_a` /
`transpose_b`. ttnn inserts a manual `ttnn.transpose` for those flags UNLESS the chosen program
config is one of the three MultiCoreReuse families (matmul.cpp:200), and the trimul pins
`MatmulMultiCoreReuseMultiCastProgramConfig`, which is in that set. So the flag should fold into
the program and the separate op should be deletable.

Three things have to hold, and only the device can say: the flagged form must RUN, it must be
bit-exact against the transpose+matmul it replaces, and it must not cost more than it saves.
"""
import json
import statistics as st
import time

import torch

torch.set_grad_enabled(False)
import ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
DRAM = ttnn.DRAM_MEMORY_CONFIG
S, C = 512, 128
SLT = S // 32
pc = T._triangle_mul_program_config(SLT)
ckc = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
print("program_config:", type(pc).__name__, "in0_block_w", pc.in0_block_w,
      "per_core_M", pc.per_core_M, "per_core_N", pc.per_core_N)

af_h = torch.randn(1, C, S, S, dtype=torch.float32).bfloat16()
bf_h = torch.randn(1, C, S, S, dtype=torch.float32).bfloat16()
af = ttnn.from_torch(af_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                     memory_config=DRAM)
bf = ttnn.from_torch(bf_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                     memory_config=DRAM)


def timed(fn, reps=4):
    ts, out = [], None
    for i in range(reps + 1):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) * 1e3
        if i:
            ts.append(dt)
        if out is None:
            out = o
        else:
            ttnn.deallocate(o)
    return st.median(ts), ts, out


def mm(a, b, **kw):
    return ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=DRAM,
                       program_config=pc, dtype=ttnn.bfloat16, **kw)


res = {}

# --- what production does today: transpose the b operand, then matmul -------------------------
def today_b():
    bt = ttnn.transpose(bf, -2, -1, memory_config=DRAM)
    o = mm(af, bt)
    ttnn.deallocate(bt)
    return o


ms_today_b, s_today_b, ref_b = timed(today_b)
ms_tr, s_tr, _tr = timed(lambda: ttnn.transpose(bf, -2, -1, memory_config=DRAM))
ttnn.deallocate(_tr)
ms_mm, s_mm, _mm_out = timed(lambda: mm(af, bf))
ttnn.deallocate(_mm_out)
print("transpose(b) alone        %8.4f ms  %s" % (ms_tr, ["%.3f" % v for v in s_tr]))
print("matmul alone (no flag)    %8.4f ms  %s" % (ms_mm, ["%.3f" % v for v in s_mm]))
print("today: transpose + matmul %8.4f ms  %s" % (ms_today_b, ["%.3f" % v for v in s_today_b]))

for label, fn, ref in (("transpose_b=True", lambda: mm(af, bf, transpose_b=True), ref_b),):
    try:
        ms, ss, out = timed(fn)
    except Exception as e:                                                      # noqa: BLE001
        print("%-18s REFUSED: %s" % (label, str(e)[:200]))
        res[label] = {"refused": str(e)[:400]}
        continue
    a = ttnn.to_torch(out).float()
    r = ttnn.to_torch(ref).float()
    eq = torch.equal(a, r)
    mx = (a - r).abs().max().item()
    print("%-18s %8.4f ms  %s  torch.equal=%s max_abs=%.6g  speedup vs today %.4fx" % (
        label, ms, ["%.3f" % v for v in ss], eq, mx, ms_today_b / ms))
    res[label] = {"ms": ms, "samples": ss, "equal": bool(eq), "max_abs": mx,
                  "ms_today": ms_today_b}
    ttnn.deallocate(out)

# --- the ending trimul wants the A operand transposed -----------------------------------------
def today_a():
    at = ttnn.transpose(af, -2, -1, memory_config=DRAM)
    o = mm(at, bf)
    ttnn.deallocate(at)
    return o


ms_today_a, s_today_a, ref_a = timed(today_a)
print("today: transpose_a + matmul %8.4f ms" % ms_today_a)
try:
    ms, ss, out = timed(lambda: mm(af, bf, transpose_a=True))
    a = ttnn.to_torch(out).float()
    r = ttnn.to_torch(ref_a).float()
    print("transpose_a=True   %8.4f ms  torch.equal=%s max_abs=%.6g  speedup %.4fx" % (
        ms, torch.equal(a, r), (a - r).abs().max().item(), ms_today_a / ms))
    res["transpose_a=True"] = {"ms": ms, "samples": ss, "equal": bool(torch.equal(a, r)),
                               "max_abs": (a - r).abs().max().item(), "ms_today": ms_today_a}
    ttnn.deallocate(out)
except Exception as e:                                                          # noqa: BLE001
    print("transpose_a=True   REFUSED: %s" % str(e)[:200])
    res["transpose_a=True"] = {"refused": str(e)[:400]}

res["transpose_alone_ms"] = ms_tr
res["matmul_alone_ms"] = ms_mm
json.dump(res, open("/home/ttuser/scratch/uod2/mm_transpose.json", "w"), indent=1)

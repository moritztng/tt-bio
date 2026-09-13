"""Is the D4 deferral worth taking on the trimul's L1 split path too?

`why_kept.py` shows the lever reaches 560 of 560 matmul calls at 512 aa and 0 of 2240 at 298 aa.
The 298 calls all go down the four-way split path, where `memory_config` is L1, the chunk is
[1, 320, 320, 32] and `_transform_chunk` does NOT decompose: the (0,3,2,1) chunk is one
`ttnn.permute` that carries the channel move and the inner swap together. There is no separate
transpose op to delete there, so deferring is a trade, not a deletion: the permute drops to
(0,3,1,2) -- which `_channel_move` can hand to the hand-written reblock kernel -- and the matmul
pays for the swap instead. Only the device can say which side is bigger.
"""
import json
import statistics as st
import time

import torch

torch.set_grad_enabled(False)
import ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
L1 = ttnn.L1_MEMORY_CONFIG
S, C = 320, 32
pc = T._triangle_mul_program_config(S // 32)
ckc = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
print("program_config:", type(pc).__name__, "in0_block_w", pc.in0_block_w,
      "per_core_M", pc.per_core_M, "per_core_N", pc.per_core_N, flush=True)
print("reblock eligible for (0,3,1,2) at this shape:", flush=True)

pre_h = torch.randn(1, S, S, C, dtype=torch.float32).bfloat16()
oth_h = torch.randn(1, S, S, C, dtype=torch.float32).bfloat16()


def up(h):
    return ttnn.from_torch(h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=L1)


pre, oth = up(pre_h), up(oth_h)
print("  eligible:", T._reblock.eligible(pre, L1), flush=True)


def timed(fn, reps=7):
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
    return ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=L1,
                       program_config=pc, dtype=ttnn.bfloat16, **kw)


res = {}


def report(label, ms, ss):
    print("%-34s %8.4f ms  %s" % (label, ms, ["%.3f" % v for v in ss]), flush=True)
    res[label] = {"ms": ms, "samples": ss}


# --- the two relayouts, alone -----------------------------------------------------------------
ms_p2131, s1, o1 = timed(lambda: ttnn.permute(pre, (0, 3, 2, 1), memory_config=L1))
ttnn.deallocate(o1)
report("permute (0,3,2,1) alone", ms_p2131, s1)
ms_move, s2, o2 = timed(lambda: T._channel_move(pre, L1))
ttnn.deallocate(o2)
report("_channel_move (0,3,1,2) alone", ms_move, s2)
ms_mm, s3, o3 = timed(lambda: mm(T._channel_move(pre, L1), T._channel_move(oth, L1)))
ttnn.deallocate(o3)
report("matmul alone (both moved)", ms_mm, s3)

# --- b is the transposed operand (starting trimul) --------------------------------------------
def today_b():
    a = T._channel_move(pre, L1)
    b = ttnn.permute(oth, (0, 3, 2, 1), memory_config=L1)
    o = mm(a, b)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    return o


def defer_b():
    a = T._channel_move(pre, L1)
    b = T._channel_move(oth, L1)
    o = mm(a, b, transpose_b=True)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    return o


ms_tb, s_tb, ref_b = timed(today_b)
report("today  move(a) + permute(b) + mm", ms_tb, s_tb)
try:
    ms_db, s_db, out_b = timed(defer_b)
    a = ttnn.to_torch(out_b).float()
    r = ttnn.to_torch(ref_b).float()
    eq, mx = torch.equal(a, r), (a - r).abs().max().item()
    report("defer  move(a) + move(b) + mm(tb)", ms_db, s_db)
    print("   -> speedup %.4fx  torch.equal=%s  max_abs=%.6g" % (ms_tb / ms_db, eq, mx), flush=True)
    res["transpose_b"] = {"today_ms": ms_tb, "defer_ms": ms_db, "speedup": ms_tb / ms_db,
                          "equal": bool(eq), "max_abs": mx}
    ttnn.deallocate(out_b)
except Exception as e:                                                          # noqa: BLE001
    print("defer_b REFUSED: %s" % str(e)[:300], flush=True)
    res["transpose_b"] = {"refused": str(e)[:400]}
ttnn.deallocate(ref_b)

# --- a is the transposed operand (ending trimul) ----------------------------------------------
def today_a():
    a = ttnn.permute(pre, (0, 3, 2, 1), memory_config=L1)
    b = T._channel_move(oth, L1)
    o = mm(a, b)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    return o


def defer_a():
    a = T._channel_move(pre, L1)
    b = T._channel_move(oth, L1)
    o = mm(a, b, transpose_a=True)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    return o


ms_ta, s_ta, ref_a = timed(today_a)
report("today  permute(a) + move(b) + mm", ms_ta, s_ta)
try:
    ms_da, s_da, out_a = timed(defer_a)
    a = ttnn.to_torch(out_a).float()
    r = ttnn.to_torch(ref_a).float()
    eq, mx = torch.equal(a, r), (a - r).abs().max().item()
    report("defer  move(a) + move(b) + mm(ta)", ms_da, s_da)
    print("   -> speedup %.4fx  torch.equal=%s  max_abs=%.6g" % (ms_ta / ms_da, eq, mx), flush=True)
    res["transpose_a"] = {"today_ms": ms_ta, "defer_ms": ms_da, "speedup": ms_ta / ms_da,
                          "equal": bool(eq), "max_abs": mx}
    ttnn.deallocate(out_a)
except Exception as e:                                                          # noqa: BLE001
    print("defer_a REFUSED: %s" % str(e)[:300], flush=True)
    res["transpose_a"] = {"refused": str(e)[:400]}

res["shape"] = [1, S, S, C]
res["permute_0321_ms"] = ms_p2131
res["channel_move_ms"] = ms_move
res["matmul_ms"] = ms_mm
json.dump(res, open("/home/ttuser/scratch/uod/mm_transpose_l1.json", "w"), indent=1)
print("WROTE /home/ttuser/scratch/uod/mm_transpose_l1.json", flush=True)

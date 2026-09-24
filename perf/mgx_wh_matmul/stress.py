#!/usr/bin/env python3
"""Large-sample rate of the Wormhole HiFi4 -2^k misses per arm, including the in1 bit split.

The triangle product [1, C, S, S] x [1, C, S, S]^T (randn, seed per batch), trimul-style 2D multicast
plan on the full grid. in1 is what the LLK loads to SrcA (llk_math_matmul.h: "in0 - loaded to SrcB,
in1 - loaded to SrcA"). Arms:

  hifi4_w8 / hifi4_w1      bf16 operands, in0_block_w 8 / 1
  hifi4_w1_f32             fp32 operands, in0_block_w 1
  hifi3_w8                 the HiFi3 trap (drops SrcA-low x SrcB-low)
  hifi4_w8_in1hi           HiFi4 on in1 with its 3 low significand bits cleared: SrcA-low is zero, so
                           the fourth phase multiplies zeros. Scored against its own float64 product
  split_w8                 in1 = hi5 + lo3 (both exact bf16, each <= 5 significant bits), two HiFi3
                           products summed in fp32: every product exact, no SrcA-low phase at all

wrong: |err| > 0.25 x sqrt(sum a^2 b^2) (the census bar); the reference is a float64 product of
the operands as the device holds them.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402
import ttnn  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kt", type=int, default=64)
ap.add_argument("--channels", type=int, default=8)
ap.add_argument("--batches", type=int, default=8)
ap.add_argument("--arms", default="hifi4_w8,hifi4_w1,hifi4_w1_f32,hifi3_w8,hifi4_w8_in1hi,split_w8")
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.set_num_threads(16)
dev = T.get_device()
kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
gx, gy = T.COMPUTE_GRID_MAIN
kt, s = a.kt, a.kt * 32
Mt, Nt = -(-kt // gy), -(-kt // gx)


def hi5(x):
    """x (bf16) with its 3 low significand bits cleared: at most 5 significant bits."""
    return (x.view(torch.int16) & ~7).view(torch.bfloat16)


def mm(ta, tb, fid, w, l1=True):
    ckc = kcls(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False, fp32_dest_acc_en=True,
               packer_l1_acc=l1)
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=w, out_subblock_h=1, out_subblock_w=1,
        out_block_h=Mt, out_block_w=Nt, per_core_M=Mt, per_core_N=Nt, transpose_mcast=False,
        fused_activation=None, fuse_batch=False)
    o = ttnn.matmul(ta, tb, transpose_b=True, compute_kernel_config=ckc, program_config=pc, dtype=ttnn.float32)
    h = ttnn.to_torch(o).double()
    ttnn.deallocate(o)
    return h


def put(x, dt):
    return ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)


tot = {arm: {"elems": 0, "wrong": 0, "max_q": 0.0, "at": []} for arm in a.arms.split(",")}
for bt in range(a.batches):
    g = torch.Generator().manual_seed(7000 + 97 * kt + bt)
    Af = torch.randn(1, a.channels, s, s, generator=g)
    Bf = torch.randn(1, a.channels, s, s, generator=g)
    A, B = Af.bfloat16(), Bf.bfloat16()
    cache = {}

    def ref_of(Ax, Bx):
        k = (id(Ax), id(Bx))
        if k not in cache:
            A64, B64 = Ax.double(), Bx.double()
            cache[k] = (A64 @ B64.transpose(-1, -2), ((A64 * A64) @ (B64 * B64).transpose(-1, -2)).sqrt())
        return cache[k]

    tA, tB = put(A, ttnn.bfloat16), put(B, ttnn.bfloat16)
    for arm in tot:
        if arm == "hifi4_w8":
            o, (ref, sc) = mm(tA, tB, "HiFi4", 8), ref_of(A, B)
        elif arm == "hifi4_w1":
            o, (ref, sc) = mm(tA, tB, "HiFi4", 1), ref_of(A, B)
        elif arm in ("nol1_w1", "nol1_w2", "nol1_w8"):
            # packer L1 accumulation off: each K block's partial is spilled and reloaded into dest
            o, (ref, sc) = mm(tA, tB, "HiFi4", int(arm[-1]), l1=False), ref_of(A, B)
        elif arm == "hifi3_w8":
            o, (ref, sc) = mm(tA, tB, "HiFi3", 8), ref_of(A, B)
        elif arm == "hifi4_w1_f32":
            # the device holds fp32 operands as tf32 in the source registers; score against the fp32
            # values (the truncation is ~1e-3 of the scale, far under the bar)
            fa, fb = put(Af, ttnn.float32), put(Bf, ttnn.float32)
            o, (ref, sc) = mm(fa, fb, "HiFi4", 1), ref_of(Af, Bf)
            ttnn.deallocate(fa)
            ttnn.deallocate(fb)
        elif arm == "hifi4_w8_in1hi":
            Bh = hi5(B)
            tb = put(Bh, ttnn.bfloat16)
            o, (ref, sc) = mm(tA, tb, "HiFi4", 8), ref_of(A, Bh)
            ttnn.deallocate(tb)
        elif arm in ("abs_w8", "abs_w1", "abspos_w8"):
            # every product non-negative (abs_*), or only in1 non-negative (abspos_w8)
            Aa = A.abs() if arm != "abspos_w8" else A
            Ba = B.abs()
            t0, t1 = put(Aa, ttnn.bfloat16), put(Ba, ttnn.bfloat16)
            o, (ref, sc) = mm(t0, t1, "HiFi4", 1 if arm == "abs_w1" else 8), ref_of(Aa, Ba)
            ttnn.deallocate(t0)
            ttnn.deallocate(t1)
        elif arm in ("sign4_w8", "sign4_w1"):
            # A = A+ - A-, B = B+ - B-: four products of non-negative operands, P - N on the host
            w = 1 if arm == "sign4_w1" else 8
            Ap, An, Bp, Bn = (put(x, ttnn.bfloat16) for x in (A.clamp(min=0), (-A).clamp(min=0),
                                                               B.clamp(min=0), (-B).clamp(min=0)))
            o = (mm(Ap, Bp, "HiFi4", w) + mm(An, Bn, "HiFi4", w)) - (mm(Ap, Bn, "HiFi4", w) + mm(An, Bp, "HiFi4", w))
            ref, sc = ref_of(A, B)
            for x in (Ap, An, Bp, Bn):
                ttnn.deallocate(x)
        elif arm in ("hi_part", "lo_part"):
            # each chunk of the split alone, against its own float64 product and its own scale
            Bh = hi5(B)
            Bp = Bh if arm == "hi_part" else (B.double() - Bh.double()).bfloat16()
            tb = put(Bp, ttnn.bfloat16)
            o, (ref, sc) = mm(tA, tb, "HiFi3", 8), ref_of(A, Bp)
            ttnn.deallocate(tb)
        elif arm == "split_w8":
            Bh = hi5(B)
            Bl = (B.double() - Bh.double()).bfloat16()
            assert torch.equal(Bh.double() + Bl.double(), B.double())
            th, tl = put(Bh, ttnn.bfloat16), put(Bl, ttnn.bfloat16)
            o, (ref, sc) = mm(tA, th, "HiFi3", 8) + mm(tA, tl, "HiFi3", 8), ref_of(A, B)
            ttnn.deallocate(th)
            ttnn.deallocate(tl)
        q = (o - ref).abs() / sc.clamp(min=1e-30)
        bad = torch.nonzero(q > 0.25)
        t = tot[arm]
        t["elems"] += ref.numel()
        t["wrong"] += int(bad.shape[0])
        t["max_q"] = max(t["max_q"], round(float(q.max()), 4))
        t["at"] += [[bt, c, i, j, round(float(ref[0, c, i, j]), 3), round(float((o - ref)[0, c, i, j]), 3)]
                    for _, c, i, j in bad[:max(0, 6 - len(t["at"]))].tolist()]
        qi = int(q.argmax())
        t["argmax"] = [bt] + [int(x) for x in torch.unravel_index(torch.tensor(qi), q.shape)] + [
            round(float(o.flatten()[qi] - ref.flatten()[qi]), 5), round(float(sc.flatten()[qi]), 4)] if float(q.max()) > t.get("_qm", 0) else t.get("argmax")
        t["_qm"] = max(t.get("_qm", 0), float(q.max()))
        t["rel_l2"] = round(float((o - ref).norm() / ref.norm()), 7)
        print(json.dumps({"batch": bt, "arm": arm, **{k: v for k, v in t.items() if k != "at"}}), flush=True)
    ttnn.deallocate(tA)
    ttnn.deallocate(tB)
print("TOTAL", json.dumps(tot), flush=True)
if a.out:
    Path(a.out).write_text(json.dumps({"arch": str(dev.arch()), "kt": kt, "rows": tot}, indent=1) + "\n")

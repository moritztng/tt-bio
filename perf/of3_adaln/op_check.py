#!/usr/bin/env python3
"""dit_adaln (adaln.adaln: layer_norm and the per-token scale / shift in one kernel) against float64 and
against the shipped pair (ttnn.layer_norm, then ttnn.addcmul), plus microseconds per call of both.

Shape: OpenFold3's token DiT at 512 tokens and one sample, a [1, 512, 768]; scale = sigmoid(..) in (0, 1),
shift ~ N(0, 1). Cases cover the dtypes the fold can hand it (bf16 everywhere; fp32 scale / shift).
Usage: TT_VISIBLE_DEVICES=<card> timeout 900 python perf/of3_adaln/op_check.py [--n 512] --out r.json
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, adaln, mm_generic as MG

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--c", type=int, default=768)
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = T.get_device()
torch.manual_seed(0)
xs = torch.randn(1, a.n, a.c) * 3 + 0.7
scs = torch.sigmoid(torch.randn(1, a.n, a.c))
sbs = torch.randn(1, a.n, a.c)


def err(t, r):
    t = t.double()
    assert torch.isfinite(t).all(), "non-finite output"
    return {"max": float((t - r).abs().max()), "rel_rms": float(((t - r) ** 2).mean().sqrt() / r.pow(2).mean().sqrt())}


def timed(f):
    f(); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter(); o = f(); ttnn.synchronize_device(dev); ts.append(time.perf_counter() - t0)
        ttnn.deallocate(o)
    return sorted(ts)[len(ts) // 2] * 1e6


res = {"n": a.n, "c": a.c, "arch": str(dev.arch()), "cases": {}}
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
for name, xd, sd in (("bf16", ttnn.bfloat16, ttnn.bfloat16), ("x_bf16_s_fp32", ttnn.bfloat16, ttnn.float32),
                     ("fp32", ttnn.float32, ttnn.float32)):
    up = lambda t, d: ttnn.from_torch(t, dtype=d, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    X, SC, SB = up(xs, xd), up(scs, sd), up(sbs, sd)
    xq, scq, sbq = (ttnn.to_torch(t).double() for t in (X, SC, SB))   # the values the device holds
    ln = (xq - xq.mean(-1, keepdim=True)) / torch.sqrt(xq.var(-1, unbiased=False, keepdim=True) + 1e-5)
    ref = sbq + ln * scq

    def shipped():
        an = ttnn.layer_norm(X, epsilon=1e-5, compute_kernel_config=ckc)
        o = ttnn.addcmul(SB, an, SC, value=1.0)
        ttnn.deallocate(an)
        return o

    def fused():
        return adaln.adaln(X, SC, SB, MG.ckc_args(ckc), tuple(T.COMPUTE_GRID_MAIN))

    case = {}
    try:
        f = fused()
        assert f is not None, f"declined: {adaln.REJECTS}"
        case["fused"] = err(ttnn.to_torch(f), ref)
        case["fused_vs_shipped_max"] = float((ttnn.to_torch(f).double() - ttnn.to_torch(shipped()).double()).abs().max())
        case["fused_us"] = timed(fused)
    except Exception as e:  # noqa: BLE001
        case["fused_error"] = repr(e)[:600]
    case["shipped"] = err(ttnn.to_torch(shipped()), ref)
    case["shipped_us"] = timed(shipped)
    res["cases"][name] = case
    print(name, json.dumps(case), flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))

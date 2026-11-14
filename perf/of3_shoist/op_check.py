#!/usr/bin/env python3
"""addcmul_at against float64 and against ttnn.addcmul on the slices it replaces, plus op timings.

The DiT shape: y = LN(a) [1, 512, 768], x / z = 768-column windows of [1, 512, 36864] wide s-term
tensors. Prints max/rel error of both paths vs float64 and the per-call device time of the kernel,
the ttnn.addcmul it replaces, one [512,384]x[384,768] linear and the wide [384, 36864] linear.
Usage: TT_VISIBLE_DEVICES=<card> timeout 600 python perf/of3_shoist/op_check.py [--rows 512] --out r.json
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, ops
from tt_bio.addcmul_at import addcmul_at, STATS

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=512)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = T.get_device()
R, C, W = a.rows, 768, 48 * 768
torch.manual_seed(0)
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
xs, ys, zs = torch.randn(1, R, W), torch.randn(1, R, C), torch.rand(1, R, W)
xs, ys, zs = (t.bfloat16() for t in (xs, ys, zs))
X, Y, Z = up(xs), up(ys), up(zs)
res = {"rows": R, "cases": []}
for blk in (0, 1, 23, 47):
    off = blk * C
    ref = xs[..., off:off + C].double() + ys.double() * zs[..., off:off + C].double()
    out = addcmul_at(X, Y, Z, C, x_off=off, z_off=off)
    assert out is not None, "declined"
    k = ttnn.to_torch(out).double()
    base = ttnn.to_torch(ttnn.addcmul(up(xs[..., off:off + C]), Y, up(zs[..., off:off + C]), value=1.0)).double()
    e = lambda t: {"max": float((t - ref).abs().max()), "rel_rms": float(((t - ref) ** 2).mean().sqrt() / ref.pow(2).mean().sqrt())}  # noqa: E731
    res["cases"].append({"slot": blk, "kernel": e(k), "ttnn_addcmul": e(base),
                         "kernel_vs_ttnn_max": float((k - base).abs().max())})
    print(res["cases"][-1], flush=True)
# Inputs untouched (a writer with a bad address writes into whatever is there).
assert torch.equal(ttnn.to_torch(X), xs.float().bfloat16()) and torch.equal(ttnn.to_torch(Z), zs)


def timed(f, n=200):
    for _ in range(5):
        f()
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(n):
        f()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t) / n * 1e6


xd, zd = up(xs[..., :C]), up(zs[..., :C])
s = up(torch.randn(1, R, 384).bfloat16())
w1, wW = up(torch.randn(384, C).bfloat16()), up(torch.randn(384, W).bfloat16())
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
res["us"] = {
    "addcmul_at": timed(lambda: ttnn.deallocate(addcmul_at(X, Y, Z, C, x_off=5 * C, z_off=5 * C))),
    "ttnn_addcmul": timed(lambda: ttnn.deallocate(ttnn.addcmul(xd, Y, zd, value=1.0))),
    "linear_384x768": timed(lambda: ttnn.deallocate(ops.linear(s, w1, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN))),
    "linear_384x36864": timed(lambda: ttnn.deallocate(ops.linear(s, wW, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)), 50),
}
res["stats"] = STATS
print(json.dumps(res["us"]), flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))

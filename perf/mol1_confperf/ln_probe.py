"""fp32_ln vs the composed fp32_layer_norm: accuracy against float64 and warm device time per call.
TT_VISIBLE_DEVICES=N python ln_probe.py OUT.json [X0.pt]
Inputs: a single-track-like x (a shared vector ~1e3x the per-token part, like Mol-1 trunk block 43) at 20, 256 and
1024 tokens, plus X0.pt (x0_dump.py's real block-43 Transition input) when given. Error metric is the whl4 one: rel
err of the per-token part (output minus its token mean) against float64."""
import json
import sys
import time

import torch

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import fp32_ln  # noqa: E402
from tt_bio.tenstorrent import fp32_layer_norm, get_device  # noqa: E402

dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
tok = lambda v: v - v.mean(0)  # noqa: E731


def ref(x, w, b):
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mu) / (var + 1e-5).sqrt() * w + b


def case(name, x, w, b):
    n, k = x.shape
    r = ref(x.double(), w.double(), b.double())
    X = ttnn.from_torch(x.float().reshape(1, n, k), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
    W = ttnn.from_torch(w.float().reshape(1, k), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    B = ttnn.from_torch(b.float().reshape(1, k), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    wd, bd = ttnn.to_torch(W).double().reshape(k), ttnn.to_torch(B).double().reshape(k)
    r = ref(x.double(), wd, bd)   # the bf16 weights both paths see
    out = {}
    arms = {"fused": lambda dt: fp32_ln.layer_norm(X, W, B, 1e-5, dt),
            "composed": lambda dt: fp32_layer_norm(X, W, B, 1e-5, ckc, dt)}
    for arm, f in arms.items():
        for dt in (ttnn.float32, ttnn.bfloat16):
            y = ttnn.to_torch(f(dt)).double().reshape(-1, k)[:n]
            tag = f"{arm}_{'f32' if dt == ttnn.float32 else 'bf16'}"
            out[f"{tag}_tok_relerr"] = float((tok(y) - tok(r)).norm() / tok(r).norm())
            out[f"{tag}_relerr"] = float((y - r).norm() / r.norm())
            out[f"{tag}_finite"] = bool(torch.isfinite(y).all())
        ttnn.synchronize_device(dev)
        reps = 200
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.deallocate(f(ttnn.bfloat16))
        t_host = time.perf_counter() - t0
        ttnn.synchronize_device(dev)
        out[f"{arm}_us_per_call"] = (time.perf_counter() - t0) / reps * 1e6
        out[f"{arm}_host_us_per_call"] = t_host / reps * 1e6
    yf = ttnn.to_torch(arms["fused"](ttnn.float32)).double().reshape(-1, k)[:n]
    yc = ttnn.to_torch(arms["composed"](ttnn.float32)).double().reshape(-1, k)[:n]
    out["fused_vs_composed_f32_maxabs"] = float((yf - yc).abs().max())
    out["fused_eq_composed_f32"] = bool(torch.equal(yf, yc))
    print(name, json.dumps(out), flush=True)
    return out


g = torch.Generator().manual_seed(0)
res = {}
k = 384
for n in (20, 256, 1024):
    shared = torch.randn(k, generator=g, dtype=torch.float64) * 150.0
    x = shared + torch.randn(n, k, generator=g, dtype=torch.float64) * 0.15
    w = 1.0 + 0.1 * torch.randn(k, generator=g, dtype=torch.float64)
    b = 0.1 * torch.randn(k, generator=g, dtype=torch.float64)
    res[f"synthetic_{n}"] = case(f"synthetic_{n}", x, w, b)
if len(sys.argv) > 2:
    d = torch.load(sys.argv[2], weights_only=False)
    res["block43"] = case("block43", d["x"].double(), d["w"].double(), d["b"].double())
res["arch"] = str(dev.arch())
json.dump(res, open(sys.argv[1], "w"), indent=1)

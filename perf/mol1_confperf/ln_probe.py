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
from tt_bio.tenstorrent import get_device  # noqa: E402

dev = get_device()
tok = lambda v: v - v.mean(0)  # noqa: E731
_PARAMS = {}


def _fp32_row_sum(x, mc):
    """Sum over the last dim (a multiple of 32) of an fp32 `x` from fp32 elementwise adds only: tile columns folded in
    halves while their count is even, the rest added in turn, then the 32 columns of one tile halved down to one.

    Wormhole's FPU reductions do not hold fp32 here: on Mol-1's trunk block 43 input `ttnn.sum` and a matmul with ones
    both land the 32-column partial of xc^2 1.07e-3 off and the layer norm's per-token part 0.50 off, where these
    adds give 1.6e-7 and 3.1e-4 against a 1.3e-4 fp32 floor (perf/mol1_whl4/ln_var_probe.py, one .111 chip). The
    tokens of a single track share a vector ~1e3x their differences, so 1/sigma needs ~1e-6 per token.
    """
    lead = list(x.shape)[:-1]
    w = x.shape[-1]
    if w % 32:
        raise ValueError(f"_fp32_row_sum: last dim {w} is not a multiple of 32")
    sl = lambda t, a, b: ttnn.slice(t, [0] * len(lead) + [a], lead + [b], memory_config=mc)
    while w > 32 and (w // 32) % 2 == 0:
        w //= 2
        x = ttnn.add(sl(x, 0, w), sl(x, w, 2 * w), memory_config=mc)
    t = sl(x, 0, 32) if w > 32 else x
    for c in range(32, w, 32):
        t = ttnn.add(t, sl(x, c, c + 32), memory_config=mc)
    w = 32
    while w > 1:
        w //= 2
        t = ttnn.add(sl(t, 0, w), sl(t, w, 2 * w), memory_config=mc)
    return t


def composed_ln(x, weight, bias, epsilon, dtype, memory_config=None):
    """tenstorrent.fp32_layer_norm before fp32_ln (main 2dba2f72e), kept as the probe's reference arm."""
    key = (id(weight), id(bias))
    params = _PARAMS.get(key)
    if params is None:
        params = _PARAMS[key] = (ttnn.typecast(weight, ttnn.float32), ttnn.typecast(bias, ttnn.float32))
    mc = memory_config
    x32 = ttnn.typecast(x, ttnn.float32, memory_config=mc) if x.dtype != ttnn.float32 else x
    inv_w = 1.0 / x.shape[-1]
    xc = ttnn.subtract(x32, ttnn.multiply(_fp32_row_sum(x32, mc), inv_w, memory_config=mc), memory_config=mc)
    if x32 is not x:
        ttnn.deallocate(x32)
    sq = ttnn.multiply(xc, xc, memory_config=mc)
    var = ttnn.multiply(_fp32_row_sum(sq, mc), inv_w, memory_config=mc)
    ttnn.deallocate(sq)
    inv = ttnn.rsqrt(ttnn.add(var, epsilon, memory_config=mc), memory_config=mc)
    ttnn.deallocate(var)
    out = ttnn.multiply_(xc, inv)
    ttnn.deallocate(inv)
    out = ttnn.add(ttnn.multiply_(out, params[0]), params[1], dtype=dtype, memory_config=mc)
    ttnn.deallocate(xc)
    return out


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
            "composed": lambda dt: composed_ln(X, W, B, 1e-5, dt)}
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

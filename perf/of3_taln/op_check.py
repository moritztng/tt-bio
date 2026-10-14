#!/usr/bin/env python3
"""triatt_ln and apb_ln (LayerNorm + q/k/v/gate/bias projection in one kernel) against float64 and against the
shipped pair (ttnn.layer_norm, then triatt_qkv.qkvgb_heads), plus the device time of both.

apb_ln (triatt_ln.ln_proj) is checked the same way against the shipped pair-bias path (ttnn.layer_norm to
DRAM, then _narrow_proj_linear) at a [128, 16] weight, the pairformer attention's 16-head pair bias.
Shape: the OF3 / OpenBind-0 pairformer triangle attention at 512 tokens, x [512, 512, 128],
4 heads of 32, the qkvgb weight [128, 4 * 128 + 32]. Prints max / rel_rms error of each path vs
float64 per output, and microseconds per call (eager, synchronized).
Usage: TT_VISIBLE_DEVICES=<card> timeout 900 python perf/of3_taln/op_check.py [--n 512] --out r.json
"""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
os.environ["TT_BIO_LEVERS"] = os.environ.get("TT_BIO_LEVERS", "") or "normal+triatt_ln+apb_ln"
import torch, ttnn
from tt_bio import tenstorrent as T, triatt_ln, triatt_qkv, mm_generic as MG

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = T.get_device()
S, K, H, C = a.n, 128, 4, 128
NB = 1                       # bias channels (OF3: 4 heads of the pair bias in one tile; values beyond are zero)
torch.manual_seed(0)
xs = (torch.randn(S, S, K) * 2 + 0.5).bfloat16()
gw, gb = (1 + 0.1 * torch.randn(K)).bfloat16(), (0.1 * torch.randn(K)).bfloat16()
w = (torch.randn(K, 4 * C + 32) / K ** 0.5)
w[:, 4 * C + 4:] = 0
w = w.bfloat16()
up = lambda t, **k: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, **k)  # noqa: E731
X, W, GW, GB = up(xs), up(w), up(gw.reshape(1, K)), up(gb.reshape(1, K))

xd = xs.double()
xn = (xd - xd.mean(-1, keepdim=True)) / torch.sqrt(xd.var(-1, unbiased=False, keepdim=True) + 1e-5) * gw.double() + gb.double()
ref = xn @ w.double()
refs = [ref[..., i * C:(i + 1) * C].reshape(S, S, H, 32).permute(0, 2, 1, 3) for i in range(4)] + [ref[..., 4 * C:4 * C + 32]]


def err(t, r):
    t = t.double()
    assert torch.isfinite(t).all(), "non-finite output"
    return {"max": float((t - r).abs().max()), "rel_rms": float(((t - r) ** 2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-30))}


def flat(o):
    (q, k, v), g, b = o
    return [q, k, v, g, b]


res = {"n": S, "cases": {}}
for fid in ("HiFi4", "HiFi2"):
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)
    def fused():
        with T.levers(os.environ["TT_BIO_LEVERS"]):   # the env var alone only names a set; this activates it
            return triatt_ln.ln_qkvgb_heads(X, GW, GB, W, MG.ckc_args(ckc), H, 32, 32, tuple(T.COMPUTE_GRID_MAIN))

    def shipped():
        xn_d = ttnn.layer_norm(X, weight=GW, bias=GB, epsilon=1e-5,
                               compute_kernel_config=T.ln_compute_kernel_config(ckc), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o = triatt_qkv.qkvgb_heads(xn_d, W, W_O, ckc, H, 32,
                                   ttnn.bfloat16, T._qkv_mm_config(xn_d, W), 32)
        ttnn.deallocate(xn_d)
        return o
    W_O = up(torch.randn(C, C).bfloat16())
    of, os_ = fused(), shipped()
    assert of is not None, ("declined", triatt_ln.REJECTS)
    case = {"fused": [err(ttnn.to_torch(t), r) for t, r in zip(flat(of), refs)]}
    case["shipped"] = None if os_ is None else [err(ttnn.to_torch(t), r) for t, r in zip(flat(os_), refs)]
    assert torch.equal(ttnn.to_torch(X), xs), "input modified"

    def timed(f, n=50):
        for _ in range(3):
            [ttnn.deallocate(t) for t in flat(f())]
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(n):
            [ttnn.deallocate(t) for t in flat(f())]
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t0) / n * 1e6
    case["us"] = {"fused": timed(fused), "shipped": None if os_ is None else timed(shipped)}
    res["cases"][fid] = case
    print(fid, json.dumps(case), flush=True)
wb = (torch.randn(K, 16) / K ** 0.5).bfloat16()
WB = up(wb)
refb = xn @ wb.double()
for fid in ("HiFi4", "HiFi2"):
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)

    def fused_b():
        with T.levers(os.environ["TT_BIO_LEVERS"]):
            return triatt_ln.ln_proj(X, GW, GB, WB, MG.ckc_args(ckc), tuple(T.COMPUTE_GRID_MAIN))

    def shipped_b():
        xn_d = ttnn.layer_norm(X, weight=GW, bias=GB, epsilon=1e-5,
                               compute_kernel_config=T.ln_compute_kernel_config(ckc), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o = T._narrow_proj_linear(xn_d, WB, ckc, ttnn.bfloat16)
        if o is None:
            o = ttnn.linear(xn_d, WB, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
        ttnn.deallocate(xn_d)
        return o
    ob, osb = fused_b(), shipped_b()
    assert ob is not None, ("apb declined", triatt_ln.REJECTS)
    case = {"fused": err(ttnn.to_torch(ob), refb), "shipped": err(ttnn.to_torch(osb), refb)}

    def timed_b(f, n=50):
        for _ in range(3):
            ttnn.deallocate(f())
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.deallocate(f())
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t0) / n * 1e6
    case["us"] = {"fused": timed_b(fused_b), "shipped": timed_b(shipped_b)}
    res["cases"]["apb_" + fid] = case
    print("apb", fid, json.dumps(case), flush=True)
res["stats"] = {"served_declined": triatt_ln.STATS, "rejects": {str(k): v for k, v in triatt_ln.REJECTS.items()}}
Path(a.out).write_text(json.dumps(res, indent=1))

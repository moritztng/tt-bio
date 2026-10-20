#!/usr/bin/env python3
"""dit_sigf32 on one DiT gate linear: sigmoid(s W + b), [1, 512, 384] x [384, 768], bf16, the DiT's HiFi2 fp32-dest config.

Arms: no activation, the wheel's "sigmoid", and `mm_sigmoid()` under dit_sigf32 (calculate_silu_f32's sigmoid via
the silu_f32 overlay). Prints per-call time (host wall over --calls back-to-back calls, synchronized) and each
output's error against float64 on the same bf16 operands, plus how many elements differ from the wheel's sigmoid.
Usage: TT_VISIBLE_DEVICES=<chip> timeout 600 python perf/of3_sigf/mm_check.py --out r.json
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T
from tt_bio.tenstorrent import CORE_GRID_MAIN

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=512)
ap.add_argument("--calls", type=int, default=400)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = T.get_device()
torch.manual_seed(0)
s = torch.randn(1, a.rows, 384).bfloat16()
w = (torch.randn(384, 768) / 384 ** 0.5).bfloat16()
b = (0.5 * torch.randn(1, 768)).bfloat16()
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
S, W, B = up(s), up(w), up(b)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
pre = s.double() @ w.double() + b.double()
ref = torch.sigmoid(pre)


def run(act):
    f = lambda: ttnn.linear(S, W, bias=B, activation=act, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)  # noqa: E731
    out = ttnn.to_torch(f()).double()
    for _ in range(20):
        ttnn.deallocate(f())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(a.calls):
        ttnn.deallocate(f())
    ttnn.synchronize_device(dev)
    return out, (time.perf_counter() - t0) / a.calls * 1e6


res = {"rows": a.rows, "calls": a.calls, "arch": "wormhole" if T.is_wormhole() else "blackhole",
       "overlay": __import__("os").environ.get("TT_BIO_METAL_OVERLAY")}
none, res["none_us"] = run(None)
res["none_max_err"] = float((none - pre).abs().max())
stock, res["sigmoid_us"] = run("sigmoid")
with T.levers("dit_sigf32"):
    act = T.mm_sigmoid()
    assert act != "sigmoid"
    f32, res["sigf32_us"] = run(act)
for k, o in (("sigmoid", stock), ("sigf32", f32)):
    res[k + "_max_err"] = float((o - ref).abs().max())
    res[k + "_rms_err"] = float((o - ref).pow(2).mean().sqrt())
res["sigf32_vs_sigmoid_max"] = float((f32 - stock).abs().max())
res["sigf32_vs_sigmoid_frac_diff"] = float((f32 != stock).double().mean())
print(json.dumps(res), flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))

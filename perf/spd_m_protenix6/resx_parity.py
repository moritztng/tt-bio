"""WS_RES_X (`x + f(LN(x))` seeded from the prologue's raw x row) against the c_20 path it replaces, and both against
float64, at Protenix-v2's --fast pair transition shape (c_z 256, bfp8 weights, one chunk, HiFi2).

Prints `bit_identical` (RES_X vs c_20, which must be True) and rel_rms of each against the float64 reference
x + fc3(silu(LN(x) w1) * (LN(x) w2)) on the dequantised weights. Usage (one chip):
    TT_VISIBLE_DEVICES=<chip> python perf/spd_m_protenix6/resx_parity.py [rows] [n] [l1acc 0|1]
"""
import json
import sys

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio import pair_swiglu_ws as W  # noqa: E402

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 64
n = int(sys.argv[2]) if len(sys.argv) > 2 else 512
l1acc = len(sys.argv) > 3 and sys.argv[3] == "1"
c, h, eps = 256, 1024, 1e-5
dev = T.get_device()
g = torch.Generator().manual_seed(0)
x = (8 * torch.randn(1, rows, n, c, generator=g)).bfloat16()
gamma = (1 + 0.1 * torch.randn(c, generator=g)).bfloat16()
beta = (0.1 * torch.randn(c, generator=g)).bfloat16()
w1, w2 = [(torch.randn(c, h, generator=g) / c ** 0.5) for _ in range(2)]
w3 = torch.randn(h, c, generator=g) / h ** 0.5
parts = W.weights(w1, w2, w3, dev, ttnn.bfloat8_b)
assert len(parts) == 1, len(parts)
w12, w3t = parts[0]
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
gt, bt = up(gamma.view(1, c)), up(beta.view(1, c))

# float64 reference on the weights the kernel holds (bfp8 dequantised)
w12d = ttnn.to_torch(w12).double()
w1d, w2d, w3d = w12d[:, :h], w12d[:, h:], ttnn.to_torch(w3t).double()
xd = x.double()
mu = xd.mean(-1, keepdim=True)
xn = (xd - mu) / ((xd - mu).pow(2).mean(-1, keepdim=True) + eps).sqrt() * gamma.double() + beta.double()
t_ref = (torch.nn.functional.silu(xn @ w1d) * (xn @ w2d)) @ w3d
ref = xd + t_ref


def run(res_x):
    W.RES_X = res_x
    xt = up(x)
    W.pair_swiglu(xt, w12, w3t, ttnn.MathFidelity.HiFi2, ln=(gt, bt, eps), l1acc=l1acc, pack_silu=True, resid=xt)
    return ttnn.to_torch(xt)


a, b = run(True), run(False)
W.RES_X = True
rel = lambda o: ((o.double() - ref).pow(2).mean().sqrt() / t_ref.pow(2).mean().sqrt()).item()  # noqa: E731
floor = rel(ref.bfloat16())
out = {"rows": rows, "n": n, "l1acc": l1acc, "bit_identical": bool(torch.equal(a, b)),
       "max_abs_diff": (a.float() - b.float()).abs().max().item(), "rel_rms_resx": rel(a), "rel_rms_c20": rel(b),
       "bf16_floor": floor, "wormhole": T.is_wormhole()}
print(json.dumps(out), flush=True)

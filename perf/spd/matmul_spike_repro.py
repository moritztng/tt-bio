#!/usr/bin/env python3
"""Minimal reproducer for a deterministic wrong element in a bf16 matmul on device.

`bc2_pairmm_precision.py` found, at seed 0, tokens 128, K=128, N=512, transpose_b: element
[0, 57, 88, 421] reads ~4.15 on every Wormhole chip against a true 0.163, in ttnn.matmul and in
minimal_matmul alike. This replays that exact case, then shrinks it to the one row of x and the one
column of w on a single 32x32 output tile and sweeps the kernel config, so the fault can be pinned to
the inputs, the fidelity, the accumulator or the packer:

    TT_VISIBLE_DEVICES=5 python perf/spd/matmul_spike_repro.py --out spike.json
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import torch  # noqa: E402
import ttnn  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()

dev = get_device()
wh = dev.arch() == ttnn.Arch.WORMHOLE_B0
Cfg = ttnn.types.WormholeComputeKernelConfig if wh else ttnn.types.BlackholeComputeKernelConfig
up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)

# Regenerate the case exactly as bc2_pairmm_precision.py draws it (bias on, tokens 128 first).
torch.manual_seed(0)
case = None
for kt, nt in [(4, 4), (4, 12), (4, 16), (4, 32), (12, 4), (16, 4), (32, 4)]:
    K, N = 32 * kt, 32 * nt
    for tb in (False, True):
        x = torch.randn(1, 128, 128, K).bfloat16().double()
        w = (torch.randn(N, K) if tb else torch.randn(K, N)).div(K ** 0.5).bfloat16().double()
        if not tb:
            torch.randn(N)
        if (K, N, tb) == (128, 512, True):
            case = x, w
x, w = case
I = (0, 57, 88, 421)
xr, wc = x[0, 57, 88], w[421]               # the row and the column that meet at the bad element
truth = float(xr @ wc)

configs = {
    "hifi4_fp32acc_l1acc": dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True),
    "hifi4_fp32acc": dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False),
    "hifi4_bf16acc": dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, packer_l1_acc=False),
    "hifi3_fp32acc": dict(math_fidelity=ttnn.MathFidelity.HiFi3, fp32_dest_acc_en=True, packer_l1_acc=False),
    "hifi2_fp32acc": dict(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=False),
    "lofi_fp32acc": dict(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False),
}
res = {"arch": str(dev.arch()), "truth": truth, "full": {}, "tile": {}, "terms": {}}
# 1. The full call, as found.
for name, kw in configs.items():
    ckc = Cfg(math_approx_mode=False, **kw)
    y = ttnn.to_torch(ttnn.matmul(up(x), up(w), transpose_b=True, compute_kernel_config=ckc)).double()
    res["full"][name] = {"value": float(y[I]), "n_bad": int(((y - x @ w.T).abs() > 0.25).sum())}
# 2. One 32x32 output tile: the bad row at row 0, the bad column at column 0, the rest zero.
X = torch.zeros(32, 128, dtype=torch.float64); X[0] = xr
W = torch.zeros(32, 128, dtype=torch.float64); W[0] = wc
for name, kw in configs.items():
    ckc = Cfg(math_approx_mode=False, **kw)
    y = ttnn.to_torch(ttnn.matmul(up(X), up(W), transpose_b=True, compute_kernel_config=ckc)).double()
    res["tile"][name] = float(y[0, 0])
# 3. Which K terms: zero all but one 32-wide K block of the row, then halve the survivors.
ckc = Cfg(math_approx_mode=False, **configs["hifi4_fp32acc_l1acc"])


def dev_dot(mask):
    Xm = X.clone(); Xm[0] = xr * mask
    y = ttnn.to_torch(ttnn.matmul(up(Xm), up(W), transpose_b=True, compute_kernel_config=ckc)).double()
    return float(y[0, 0]), float((xr * mask) @ wc)


for blk in range(4):
    m = torch.zeros(128, dtype=torch.float64); m[32 * blk:32 * blk + 32] = 1
    got, want = dev_dot(m)
    res["terms"][f"k{32 * blk}-{32 * blk + 31}"] = {"device": got, "truth": want}
lo, hi = 0, 128
while hi - lo > 1:                             # narrow to the smallest K range that still breaks
    mid = (lo + hi) // 2
    m = torch.zeros(128, dtype=torch.float64); m[lo:mid] = 1
    got, want = dev_dot(m)
    if abs(got - want) > 0.25:
        hi = mid
        continue
    m = torch.zeros(128, dtype=torch.float64); m[mid:hi] = 1
    got, want = dev_dot(m)
    if abs(got - want) > 0.25:
        lo = mid
        continue
    break
res["narrowest"] = {"k": [lo, hi], "x": xr[lo:hi].tolist(), "w": wc[lo:hi].tolist()}
m = torch.zeros(128, dtype=torch.float64); m[lo:hi] = 1
res["narrowest"]["device"], res["narrowest"]["truth"] = dev_dot(m)
# 4. Greedy: drop single K terms while the device still disagrees, down to a minimal set.
keep = m.clone()
for k in range(lo, hi):
    trial = keep.clone(); trial[k] = 0
    got, want = dev_dot(trial)
    if abs(got - want) > 0.25:
        keep = trial
ks = [int(k) for k in keep.nonzero().flatten()]
got, want = dev_dot(keep)
res["minimal"] = {"k": ks, "x": [float(xr[k]) for k in ks], "w": [float(wc[k]) for k in ks],
                  "device": got, "truth": want}
print(json.dumps(res, indent=1))
pathlib.Path(a.out).write_text(json.dumps(res, indent=1))

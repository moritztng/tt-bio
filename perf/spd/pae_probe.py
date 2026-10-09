"""Where PaeBins.on_device's pTM bias comes from: the device softmax or the fp32 FPU matmul.

ab26 (t24) read pTM/ipTM 0.002-0.004 LOW on every sample against the host path, coordinates bit-identical.
At OpenFold3 c730's PAE shape this compares, against float64: E[TM] (n=730) and the PAE expectation for
  soft  = device softmax, host float64 reduction        (isolates the softmax)
  mm    = host float64 softmax, device fp32 matmul        (isolates the matmul)
  full  = PaeBins.on_device as shipped in t24
  split = device softmax, probs and weights split hi/lo in bf16, three matmuls (bf16x3)
Prints mean signed and max |error| of E[TM] per pair and of the pTM-like row-mean max.

usage: TT_VISIBLE_DEVICES=N timeout 600 python perf/spd/pae_probe.py
"""
import torch
import ttnn

import tt_bio.tenstorrent as T
from tt_bio.protenix import PaeBins, tm_per_bin

dev = T.get_device()
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
N = 730
C = (torch.arange(64, dtype=torch.float64) + 0.5) * 0.5
torch.manual_seed(0)
# PAE-like logits: a peaked bin per pair plus noise.
peak = torch.randint(0, 64, (N, N, 1))
logits = (-(torch.arange(64).view(1, 1, 64) - peak).abs().double() * 0.6 + torch.randn(N, N, 64, dtype=torch.float64))
cols = torch.stack([C, tm_per_bin(N, C)], 1)                    # [64, 2]: pae, E[TM]
ref = torch.softmax(logits, -1) @ cols


def up(x, dt=ttnn.float32):
    return ttnn.from_torch(x.float().unsqueeze(0), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)


def down(x):
    return torch.Tensor(ttnn.to_torch(x)).double().reshape(N, N, -1)


def report(name, got):
    for j, what in enumerate(("pae", "etm")):
        e = got[..., j] - ref[..., j]
        print(f"{name:6s} {what}: mean {e.mean():+.3e} max|e| {e.abs().max():.3e}", end="   ")
    print(f"ptm-like {got[..., 1].mean(-1).max() - ref[..., 1].mean(-1).max():+.3e}")


sm = lambda x: ttnn.softmax(x, dim=-1, compute_kernel_config=ckc, numeric_stable=True)
p_dev = down(sm(up(logits)))
report("soft", p_dev @ cols)
W = lambda w: ttnn.from_torch(w.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
mm = lambda a, w: ttnn.matmul(a, W(w), compute_kernel_config=ckc)
report("mm", down(mm(up(torch.softmax(logits, -1)), cols)))
b = PaeBins.on_device(up(logits), [N], C.float(), ckc)
report("full", torch.stack([b.pae().double(), b.etm(N).double()], -1))
p = sm(up(logits))
ph = ttnn.typecast(ttnn.typecast(p, ttnn.bfloat16), ttnn.float32)
pl = ttnn.subtract(p, ph)
wh = cols.float().bfloat16().float()
wl = cols.float() - wh
report("split", down(ttnn.add(ttnn.add(mm(ph, wh), mm(pl, wh)), mm(ph, wl))))

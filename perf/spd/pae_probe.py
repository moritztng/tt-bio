"""Where PaeBins.on_device's pTM bias comes from: the device softmax or the fp32 FPU matmul.

ab26 (t24) read pTM/ipTM 0.002-0.004 LOW on every sample against the host path, coordinates bit-identical.
At OpenFold3 c730's PAE shape this compares, against float64, E[TM] (n=730) and the PAE expectation for
  fused = device ttnn.softmax, device fp32 matmul (t24, the biased form)
  mm    = host float64 softmax, device fp32 matmul (the matmul's own share)
  ratio = PaeBins.on_device after pp26: device exp(x - max) only, bf16x3 matmul against [1 | cols], host divide
on two logit sets: a soft synthetic one and a sharp one (slope 3, offset +20, the range a confident head writes).
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
peak = torch.randint(0, 64, (N, N, 1))
dist = (torch.arange(64).view(1, 1, 64) - peak).abs().double()
SETS = {"soft": -dist * 0.6 + torch.randn(N, N, 64, dtype=torch.float64),
        "sharp": 20.0 - dist * 3.0 + torch.randn(N, N, 64, dtype=torch.float64)}
cols = torch.stack([C, tm_per_bin(N, C)], 1)                    # [64, 2]: pae, E[TM]


def up(x, dt=ttnn.float32):
    return ttnn.from_torch(x.float().unsqueeze(0), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)


def down(x):
    return torch.Tensor(ttnn.to_torch(x)).double().reshape(N, N, -1)


def report(name, got):
    for j, what in enumerate(("pae", "etm")):
        e = got[..., j] - ref[..., j]
        print(f"{name:6s} {what}: mean {e.mean():+.3e} max|e| {e.abs().max():.3e}", end="   ")
    print(f"ptm-like {got[..., 1].mean(-1).max() - ref[..., 1].mean(-1).max():+.3e}", flush=True)


sm = lambda x: ttnn.softmax(x, dim=-1, compute_kernel_config=ckc, numeric_stable=True)
W = lambda w: ttnn.from_torch(w.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
mm = lambda a, w: ttnn.matmul(a, W(w), compute_kernel_config=ckc)
for tag, logits in SETS.items():
    print(f"== {tag}")
    ref = torch.softmax(logits, -1) @ cols
    report("fused", down(mm(sm(up(logits)), cols)))
    report("mm", down(mm(up(torch.softmax(logits, -1)), cols)))
    b = PaeBins.on_device(up(logits), [N], C.float(), ckc)
    report("ratio", torch.stack([b.pae().double(), b.etm(N).double()], -1))

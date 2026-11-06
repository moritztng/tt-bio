"""Per-call time and output of tt_bio.tenstorrent.fp32_single_transition on one Wormhole chip, at OpenDDE's single-track
shape (N tokens x 384, fc 384->1536 twice, 1536->384), fixed seeded input and weights so two trees can be compared:
TT_VISIBLE_DEVICES=C python fst_bench.py N OUT.pt  (OUT.pt holds the output and the warm per-call ms)."""
import sys, time
import torch
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
import tt_bio.tenstorrent as T

N, out = int(sys.argv[1]), sys.argv[2]
g = torch.Generator().manual_seed(0)
r = lambda *s, sc=1.0: torch.randn(*s, generator=g) * sc
x, lw, lb = r(1, N, 384, sc=3.0) + r(384, sc=5.0), 1 + r(384, sc=0.1), r(384, sc=0.1)
w1, w2, w3 = r(384, 1536, sc=384 ** -.5), r(384, 1536, sc=384 ** -.5), r(1536, 384, sc=1536 ** -.5)
dev = T.get_device()
up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
X, LW, LB, W1, W2, W3 = map(up, (x, lw, lb, w1, w2, w3))
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=False)
f = lambda: T.fp32_single_transition(X, LW, LB, W1, W2, W3, ckc, ttnn.bfloat16)
y = f()
for _ in range(20):
    ttnn.deallocate(f())
ttnn.synchronize_device(dev)
ms = []
for _ in range(5):
    t = time.perf_counter()
    for _ in range(50):
        ttnn.deallocate(f())
    ttnn.synchronize_device(dev)
    ms.append((time.perf_counter() - t) / 50 * 1e3)
torch.save({"y": ttnn.to_torch(y), "ms": ms, "N": N}, out)
print("N", N, "ms/call", [round(m, 4) for m in ms])

"""Device precision probe (qb2 p300c, chip pinned via TT_VISIBLE_DEVICES; see state/rfd4-vae.md for the readings)."""
import torch
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
dev = ttnn.open_device(device_id=0)
ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
x = (torch.randn(1, 256, 768, dtype=torch.float64) * 5 + 3).float().double()
ref = x.sum(-1, keepdim=True); absum = x.abs().sum(-1, keepdim=True)
ones = up(torch.ones(768, 1), ttnn.bfloat16)
X = up(x.float(), ttnn.float32)
s0 = ttnn.matmul(X, ones, dtype=ttnn.float32, compute_kernel_config=ck)
xh = x.to(torch.bfloat16); r1 = x - xh.double(); xm = r1.to(torch.bfloat16); xl = (r1 - xm.double()).to(torch.bfloat16)
parts = [ttnn.matmul(up(t, ttnn.bfloat16), ones, dtype=ttnn.float32, compute_kernel_config=ck) for t in (xh, xm, xl)]
s3 = ttnn.add(ttnn.add(parts[0], parts[1]), parts[2])
s2 = ttnn.add(parts[0], parts[1])
f = lambda s: float(((ttnn.to_torch(s).double() - ref).abs() / absum).max())
print("red fp32x", f(s0), "hi", f(parts[0]), "hi+mid", f(s2), "hi+mid+lo", f(s3))

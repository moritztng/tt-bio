"""Device precision probe (qb2 p300c, chip pinned via TT_VISIBLE_DEVICES; see state/rfd4-vae.md for the readings)."""
import torch
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
dev = ttnn.open_device(device_id=0)
ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
for e in (8, 11, 12, 14, 16, 20):
    x = torch.zeros(32, 32); x[:, 0] = 1.0; x[:, 1] = 2.0 ** -e
    w = torch.zeros(32, 32); w[0, :] = 1.0; w[1, :] = 1.0
    y = ttnn.to_torch(ttnn.matmul(up(x.bfloat16(), ttnn.bfloat16), up(w.bfloat16(), ttnn.bfloat16), dtype=ttnn.float32, compute_kernel_config=ck))
    a = ttnn.to_torch(ttnn.add(up(x[:, :1].contiguous().float(), ttnn.float32), up(x[:, 1:2].contiguous().float(), ttnn.float32)))
    print("e", e, "matmul 1+2^-e - 1 =", float(y[0, 0] - 1.0), "want", 2.0 ** -e, "| eltwise add:", float(a[0, 0] - 1.0))

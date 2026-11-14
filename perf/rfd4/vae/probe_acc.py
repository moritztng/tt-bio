"""Device precision probe (qb2 p300c, chip pinned via TT_VISIBLE_DEVICES; see state/rfd4-vae.md for the readings)."""
import torch
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
dev = ttnn.open_device(device_id=0)
torch.manual_seed(0)
x = (torch.randn(1, 128, 256, dtype=torch.float64) * 3 + 50).to(torch.bfloat16)
w = (torch.randn(256, 256, dtype=torch.float64) / 16).to(torch.bfloat16)
ref = x.double() @ w.double()
def e(y): return float((y.double() - ref).abs().max() / ref.abs().max())
H = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
W = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
for fid in ("HiFi4", "HiFi2", "LoFi"):
  for l1 in (True, False):
    for dst in (True, False):
        ck = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False, fp32_dest_acc_en=dst, packer_l1_acc=l1)
        y = ttnn.to_torch(ttnn.linear(H, W, dtype=ttnn.float32, compute_kernel_config=ck))
        y2 = ttnn.to_torch(ttnn.matmul(H, W, dtype=ttnn.float32, compute_kernel_config=ck))
        print("cfg", fid, "l1acc", l1, "fp32dst", dst, "linear", e(y), "matmul", e(y2))

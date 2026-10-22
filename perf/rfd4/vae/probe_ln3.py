"""Device precision probe (qb2 p300c, chip pinned via TT_VISIBLE_DEVICES; see state/rfd4-vae.md for the readings)."""
import torch
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
import torch.nn.functional as F
dev = ttnn.open_device(device_id=0)
ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
print("has_cfg", hasattr(ttnn, "LayerNormDefaultProgramConfig"))
import inspect
try: print(ttnn.LayerNormDefaultProgramConfig.__init__.__doc__[:400])
except Exception as e: print("doc err", e)
torch.manual_seed(0)
up = lambda t: ttnn.from_torch(t.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
x = torch.randn(1, 256, 768, dtype=torch.float64) * 5 + 3
w = 1 + 0.3 * torch.randn(768, dtype=torch.float64); b = 0.1 * torch.randn(768, dtype=torch.float64)
ref = F.layer_norm(x, (768,), w, b, 1e-5)
e = lambda y: float((ttnn.to_torch(y).double() - ref).abs().max() / ref.abs().max())
grid = dev.compute_with_storage_grid_size()
crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
recip = ttnn.create_layer_norm_reciprocals(dev, crs, 768)
for kw in ({}, {"use_welford": True}, {"use_welford": True, "legacy_rsqrt": True}):
    try:
        pc = ttnn.LayerNormDefaultProgramConfig(**kw)
        y = ttnn.layer_norm(up(x), epsilon=1e-5, weight=up(w.reshape(1, -1)), bias=up(b.reshape(1, -1)), compute_kernel_config=ck, program_config=pc, **({"recip_tensor": recip} if kw.get("use_welford") else {}))
        print("cfg", kw, e(y))
    except Exception as ex:
        print("cfg", kw, "ERR", str(ex)[:150])

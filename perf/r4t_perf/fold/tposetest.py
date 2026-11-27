import torch
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
from tt_bio import tenstorrent as tt
import ttnn
dev = tt.get_device()
torch.manual_seed(0)
x = torch.randn(384, 1024) * 1e-3
t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
for name, f in [("transpose", lambda a: ttnn.transpose(a, -2, -1)),
                ("rm_transpose", lambda a: ttnn.transpose(ttnn.to_layout(a, ttnn.ROW_MAJOR_LAYOUT), -2, -1))]:
    y = ttnn.to_torch(f(t)).reshape(1024, 384)
    d = (y != x.t())
    print(name, "differ", int(d.sum()), "of", d.numel(), "max", float((y - x.t()).abs().max()),
          "bf16-exact?", bool(torch.equal(y, x.t().to(torch.bfloat16).float())))

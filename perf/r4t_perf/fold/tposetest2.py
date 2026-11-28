import torch
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
from tt_bio import tenstorrent as tt
import ttnn
dev = tt.get_device()
torch.manual_seed(0)
x = torch.randn(384, 1024) * 1e-3
t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
import time
for name, f in [("permute_tile", lambda a: ttnn.permute(a, (1, 0))),
                ("permute_rm", lambda a: ttnn.permute(ttnn.to_layout(a, ttnn.ROW_MAJOR_LAYOUT), (1, 0)))]:
    try:
        y = ttnn.to_torch(f(t)).reshape(1024, 384)
        print(name, "differ", int((y != x.t()).sum()), "max", float((y - x.t()).abs().max()))
    except Exception as e:
        print(name, "ERR", str(e)[:150])

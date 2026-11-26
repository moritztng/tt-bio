import torch, time
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
from tt_bio import tenstorrent as tt
import ttnn
dev = tt.get_device()
torch.manual_seed(0)
x = torch.randn(384, 1024) * 1e-3
t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
print("has bitcast", hasattr(ttnn, "bitcast"))
try:
    u = ttnn.bitcast(t, ttnn.uint32)
    y = ttnn.to_torch(ttnn.transpose(u, -2, -1))
    y = y.to(torch.int64).to(torch.int32).view(torch.float32).reshape(1024, 384) if y.dtype != torch.float32 else y.reshape(1024, 384)
    print("bitcast_transpose differ", int((y != x.t()).sum()), y.dtype)
    ttnn.synchronize_device(dev); a = time.perf_counter()
    for _ in range(10): z = ttnn.transpose(u, -2, -1)
    ttnn.synchronize_device(dev); print("t_ms", (time.perf_counter() - a) * 100)
except Exception as e:
    print("ERR", str(e)[:200])

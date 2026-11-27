import time
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T
dev = T.get_device()
g = torch.Generator().manual_seed(0)
xs = [torch.randn(1024, 1280, generator=g) * 0.02 for _ in range(40)]
for src, td in (("f32", ttnn.float32), ("bf16", ttnn.bfloat16)):
    t0 = time.time(); ds = [ttnn.from_torch(x if src == "f32" else x.to(torch.bfloat16), dtype=td, layout=ttnn.TILE_LAYOUT, device=dev) for x in xs]; ttnn.synchronize_device(dev); t1 = time.time()
    ys = [ttnn.typecast(d, ttnn.bfloat8_b) for d in ds]; ttnn.synchronize_device(dev); t2 = time.time()
    ys2 = [ttnn.typecast(d, dtype=ttnn.bfloat8_b) for d in ds]; ttnn.synchronize_device(dev)
    y = ys[0]; ref = ttnn.to_torch(ttnn.from_torch(xs[0], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)).float()
    print(src, "upload %.3f typecast %.3f" % (t1 - t0, t2 - t1), "out dtype", y.dtype, ys2[0].dtype, "storage", y.storage_type(),
          "bitid", (ttnn.to_torch(y).float() == ref).float().mean().item(), (ttnn.to_torch(ys2[0]).float() == ref).float().mean().item(), flush=True)

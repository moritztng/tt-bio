"""Readback of an fp32 TILE gradient: host untilize (to_torch) vs card untilize of the uint32 bits + RM readback. Exactness + time."""
import time, torch
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
from tt_bio import tenstorrent as T
torch.set_num_threads(2)
dev = T.get_device()
def host(g): return ttnn.to_torch(g)
def card(g):
    u = ttnn.to_layout(ttnn.bitcast(g, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
    r = ttnn.to_torch(u).view(torch.float32)
    ttnn.deallocate(u); return r
for shape in ((3072, 768), (768, 768), (256, 6144), (1, 1, 384, 384)):
    x = torch.randn(shape)
    g = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    res = {}
    for name, f in (("host", host), ("card", card)) * 2:
        ttnn.synchronize_device(dev); t = time.perf_counter()
        for _ in range(10): r = f(g)
        ttnn.synchronize_device(dev); res[name] = (time.perf_counter() - t) / 10
        assert torch.equal(r.reshape(shape), x), name
    print(shape, {k: round(v * 1e3, 2) for k, v in res.items()}, "ms, both exact")

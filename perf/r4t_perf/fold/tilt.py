import time, torch
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
from tt_bio import tenstorrent as tt
import ttnn
dev = tt.get_device(); torch.manual_seed(0)
x = torch.randn(3072, 768)
def host(): return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
def card():
    r = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    t = ttnn.to_layout(r, ttnn.TILE_LAYOUT); ttnn.deallocate(r); return t
a, b = ttnn.to_torch(host()), ttnn.to_torch(card())
print("exact", torch.equal(a, b), "vs torch bf16", torch.equal(a, x.bfloat16()))
for f in (host, card, host, card):
    ttnn.synchronize_device(dev); t0 = time.perf_counter()
    for _ in range(10): ttnn.deallocate(f())
    ttnn.synchronize_device(dev); print(f.__name__, round((time.perf_counter() - t0) / 10 * 1e3, 2), "ms per 2.36 Melem")

"""Is the rows collapse a view? For a fresh tile tensor, and for esmfold2's `outer`, which
comes out of a 5-D permute and a reshape (esmfold2.py:1207-1210)."""
import sys, time
import torch, ttnn
from tt_bio.tenstorrent import get_device

dev = get_device()
T = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def probe(name, x):
    s = [int(d) for d in x.shape]
    ttnn.synchronize_device(dev); t0 = time.perf_counter()
    for _ in range(20):
        y = ttnn.reshape(x, [s[0] * s[1] * s[2], s[3]])
    ttnn.synchronize_device(dev)
    us = (time.perf_counter() - t0) / 20 * 1e6
    print(name, s, "padded", [int(d) for d in x.padded_shape], x.memory_config().memory_layout,
          "view" if y.buffer_address() == x.buffer_address() else "COPY", f"{us:.1f} us")


probe("fresh", T(torch.randn(1, 32, 512, 1024)))
prod = T(torch.randn(1, 32 * 32, 512 * 32))
p = ttnn.permute(ttnn.reshape(prod, (1, 32, 32, 512, 32)), (0, 1, 3, 2, 4))
outer = ttnn.reshape(p, (1, 32, 512, 1024))
probe("outer", outer)

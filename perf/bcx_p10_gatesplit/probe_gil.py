"""Does ttnn's op dispatch hold the GIL? Time a ttnn enqueue loop alone, a pure-Python loop
alone, then both on two threads. If dispatch released the GIL, the two would overlap."""
import threading
import time
import torch
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
from tt_bio import tenstorrent

dev = tenstorrent.get_device()
x = ttnn.from_torch(torch.randn(1, 64, 64), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)


def enq(k=30000):
    t = x
    for _ in range(k):
        t = ttnn.add(t, x)
    ttnn.synchronize_device(dev)


def py(k=30_000_000):
    s = 0
    for i in range(k):
        s += i


enq(200)
for name, fn in (("enqueue", enq), ("python", py)):
    t0 = time.perf_counter(); fn(); print(f"{name} alone {time.perf_counter()-t0:.3f}s")
res = {}
def timed(n, fn):
    t0 = time.perf_counter(); fn(); res[n] = time.perf_counter() - t0
ts = [threading.Thread(target=timed, args=a) for a in (("enqueue", enq), ("python", py))]
t0 = time.perf_counter(); [t.start() for t in ts]; [t.join() for t in ts]
print(f"together wall {time.perf_counter()-t0:.3f}s  " + "  ".join(f"{k} {v:.3f}s" for k, v in res.items()))

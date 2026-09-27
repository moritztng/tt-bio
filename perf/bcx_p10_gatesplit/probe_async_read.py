"""Does a non-blocking readback + event on this ttnn return before the device finishes, and
does the host copy it produces equal a blocking read? Also: does the event wait for exactly
this work and not for work enqueued after it."""
import time
import torch
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
from tt_bio import tenstorrent

dev = tenstorrent.get_device()
x = ttnn.from_torch(torch.randn(1, 2048, 2048), layout=ttnn.TILE_LAYOUT, device=dev,
                    dtype=ttnn.bfloat16)

def work(t, k):
    for _ in range(k):
        t = ttnn.matmul(t, x)
        t = ttnn.multiply(t, 1e-3)
    return t

ttnn.synchronize_device(dev)
for k in (1, 40):
    t0 = time.perf_counter()
    y = work(x, k)
    t1 = time.perf_counter()
    h = ttnn.from_device(y, blocking=False)
    ev = ttnn.record_event(dev)
    t2 = time.perf_counter()
    ttnn.event_synchronize(ev)
    t3 = time.perf_counter()
    a = ttnn.to_torch(h)
    b = ttnn.to_torch(y)                  # blocking reference read
    print(f"k={k} enqueue {t1-t0:.4f}s  issue-read+event {t2-t1:.4f}s  wait {t3-t2:.4f}s  "
          f"equal={torch.equal(a, b)}")

# the event must not wait for work enqueued AFTER it
ttnn.synchronize_device(dev)
y = work(x, 300)
h = ttnn.from_device(y, blocking=False)
ev = ttnn.record_event(dev)
z = work(x, 600)                            # the other trajectory's enqueue
t0 = time.perf_counter(); ttnn.event_synchronize(ev); t1 = time.perf_counter()
ttnn.synchronize_device(dev); t2 = time.perf_counter()
print(f"event wait {t1-t0:.4f}s  then rest {t2-t1:.4f}s  equal={torch.equal(ttnn.to_torch(h), ttnn.to_torch(y))}")

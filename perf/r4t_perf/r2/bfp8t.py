import os, time
from concurrent.futures import ThreadPoolExecutor
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T
dev = T.get_device()
g = torch.Generator().manual_seed(0)
xs = [torch.randn(1024, 1280, generator=g) for _ in range(40)]
M = 40 * 1024 * 1280 / 1e6
ref = [ttnn.to_torch(ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)) for x in xs[:3]]
t0 = time.time(); hs = [ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT) for x in xs]; t1 = time.time()
ds = [ttnn.to_device(h, dev) for h in hs]; ttnn.synchronize_device(dev); t2 = time.time()
print("serial: host pack %.3f s, upload %.3f s (%.0f Melem/s total)" % (t1 - t0, t2 - t1, M / (t2 - t0)), flush=True)
for nt in (2, 4, 8):
    with ThreadPoolExecutor(nt) as ex:
        t0 = time.time()
        hs = list(ex.map(lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT), xs)); t1 = time.time()
    ds = [ttnn.to_device(h, dev) for h in hs]; ttnn.synchronize_device(dev); t2 = time.time()
    same = all(torch.equal(ttnn.to_torch(ds[i]), ref[i]) for i in range(3))
    print("threads %d: host pack %.3f s, upload %.3f s (%.0f Melem/s total) bit-identical %s" % (nt, t1 - t0, t2 - t1, M / (t2 - t0), same), flush=True)
print("nproc", os.cpu_count(), "load", open("/proc/loadavg").read().split()[:3])

"""ttnn.zeros cost against shape: is the size-1 tile-padded axis the cause, and is it host-built."""
import json
import sys
import time

sys.path.insert(0, ".")
import ttnn  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

card = int(sys.argv[1]) if len(sys.argv) > 1 else 3
dev = get_device()
out = []
for shape in ([128, 128, 1, 128], [128, 128, 128], [128, 1, 128, 128], [128, 128, 32, 128],
              [256, 256, 1, 128], [256, 256, 128], [256, 1, 256, 128], [256, 256, 32, 128]):
    ts, cs = [], []
    for i in range(4):
        ttnn.synchronize_device(dev)
        t0, c0 = time.time(), time.process_time()
        z = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        ttnn.synchronize_device(dev)
        ts.append(time.time() - t0)
        cs.append(time.process_time() - c0)
        ttnn.deallocate(z)
    r = {"shape": shape, "wall_s": min(ts[1:]), "cpu_s": min(cs[1:])}
    out.append(r)
    print(r, flush=True)

json.dump(out, open("perf/bcx_afgrad/zerosdiag.json", "w"), indent=1)

"""Host tilize vs device tilize for the large one-off transfers of a Protenix-v2 c730 fold.

Upload: `from_torch(x, TILE)` tilizes on the host while holding the GIL. The alternative uploads row-major
(pre-cast in torch) and tilizes on the chip with `to_layout`. Download: `to_torch` of a TILE tensor
untilizes on the host; the alternative untilizes on the chip first. Both are permutations, so the
probe checks the results bit for bit and reports host-blocked time and time to a synced device.

    TT_VISIBLE_DEVICES=<chip> OMP_NUM_THREADS=2 python perf/spd_hostlap/tilize_probe.py
"""
import os, time
if not os.environ.get("TT_VISIBLE_DEVICES"):
    raise SystemExit("pin one chip with TT_VISIBLE_DEVICES first")
import torch
import ttnn
from tt_bio.tenstorrent import get_device
dev = get_device()
torch.manual_seed(0)
print(f"threads {torch.get_num_threads()}", flush=True)

TD = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}
CASES = [  # name, shape, ttnn dtype
    ("dit_z LN(pair_z) fp32", (1, 730, 730, 256), ttnn.float32),
    ("relp fp32", (730, 730, 139), ttnn.float32),
    ("msa feature bf16", (1, 9947, 736, 34), ttnn.bfloat16),
    ("template te_at bf16", (1, 736, 736, 108), ttnn.bfloat16),
    ("z_base bf16", (1, 730, 730, 256), ttnn.bfloat16),
    ("odd small fp32", (5919, 3), ttnn.float32),
]


def best(fn, reps=3):
    out, b, bs = None, 1e9, 1e9
    for _ in range(reps):
        if out is not None:
            ttnn.deallocate(out)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter(); out = fn(); t1 = time.perf_counter()
        ttnn.synchronize_device(dev); t2 = time.perf_counter()
        b, bs = min(b, t1 - t0), min(bs, t2 - t0)
    return out, b, bs


def up_host(x, dt):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)


def up_dev(x, dt):
    rm = ttnn.from_torch(x.to(TD[dt]), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=dt)
    t = ttnn.to_layout(rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(rm)
    return t


def down_dev(t):
    rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    h = ttnn.to_torch(rm)
    ttnn.deallocate(rm)
    return h


for name, shape, dt in CASES:
    x = torch.randn(shape)
    try:
        a, ha, sa = best(lambda: up_host(x, dt))
        b, hb, sb = best(lambda: up_dev(x, dt))
        ra, rb = ttnn.to_torch(a), ttnn.to_torch(b)
        same = ra.shape == rb.shape and torch.equal(ra, rb)
        print(f"UP   {name:<24} host-tilize host {ha:6.3f} synced {sa:6.3f} | device-tilize host {hb:6.3f} synced {sb:6.3f}"
              f" | bit-equal {same}", flush=True)
        t0 = time.perf_counter(); d1 = ttnn.to_torch(a); t1 = time.perf_counter()
        d2 = down_dev(a); t2 = time.perf_counter()
        d1b = ttnn.to_torch(a); t3 = time.perf_counter()
        d2b = down_dev(a); t4 = time.perf_counter()
        same = d1.shape == d2.shape and torch.equal(d1, d2)
        print(f"DOWN {name:<24} host-untilize {min(t1 - t0, t3 - t2):6.3f} | device-untilize {min(t2 - t1, t4 - t3):6.3f}"
              f" | bit-equal {same}", flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)
    except Exception as e:  # report and go on: one refused shape must not hide the others
        print(f"ERR  {name:<24} {type(e).__name__}: {str(e)[:300]}", flush=True)
print("done", flush=True)

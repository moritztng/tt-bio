"""Per-site bit-exactness AND per-call saving at the exact shapes+dtypes the census recorded.

The census (perf/eltwise_fusion/censusdt298_*.json) showed both models run these sites
mostly in FLOAT32, which killed the "openfold3 moves because it is bf16" hypothesis: the
two models differ in which SITE they run, not in dtype. So each site is screened on its own
here, at its own shape and dtype, against the two-op chain it replaces.
"""
import os, time, json
import torch, ttnn
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor(None, int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
from tt_bio.tenstorrent import get_device

DEV = get_device()
REPS = int(os.environ.get("REPS", "16"))
WARMUP = int(os.environ.get("WARMUP", "3"))
F32, BF16 = ttnn.float32, ttnn.bfloat16


def dv(t, dt):
    return ttnn.from_torch(t.to(torch.float32), dtype=dt, layout=ttnn.TILE_LAYOUT,
                           device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def one(fn):
    ttnn.synchronize_device(DEV)
    t0 = time.perf_counter()
    o = fn()
    ttnn.synchronize_device(DEV)
    d = time.perf_counter() - t0
    ttnn.deallocate(o)
    return d


# (label, calls, kind, big shape, small shape, scale, dtype) -- straight from the census
SITES = [
    ("px2 tenstorrent.py:7850", 1200, "tst", (1, 16, 298, 298), (1, 16, 298, 298), 48 ** -0.5, F32),
    ("px2 protenix.py:570",      300, "tst", (75, 4, 32, 128),  (75, 4, 32, 128),  32 ** -0.5, F32),
    ("px2 protenix.py:570 bf16",   3, "tst", (75, 4, 32, 128),  (75, 4, 32, 128),  32 ** -0.5, BF16),
    ("of3 dit.py:202",          1200, "tst", (1, 16, 320, 320), (1, 16, 320, 320), 48 ** -0.5, F32),
    ("of3 atomtx.py:180",        300, "tst", (1, 75, 4, 32, 128), (1, 75, 1, 32, 128), 32 ** -0.5, F32),
    ("of3 atomtx.py:180 bf16",     3, "tst", (1, 75, 4, 32, 128), (1, 75, 1, 32, 128), 32 ** -0.5, BF16),
    ("of3 dit.py:241",          1200, "ttt", (1, 320, 768),     (1, 320, 1),       None, F32),
    ("of3 atomtx.py:213",        300, "ttt", (1, 2400, 128),    (1, 2400, 1),      None, F32),
    ("of3 atomtx.py:213 bf16",     3, "ttt", (1, 2400, 128),    (1, 2400, 1),      None, BF16),
    ("of3 dmod.py:343",            1, "ttt", (1, 75, 32, 128, 16), (1, 75, 32, 128, 1), None, F32),
]

rows = []
for label, calls, kind, s1, s2, scale, dt in SITES:
    torch.manual_seed(0)
    a = dv(torch.randn(*s1, dtype=torch.float64) * 8.0, dt)
    b = dv(torch.randn(*s2, dtype=torch.float64) * 2.0, dt)
    c = dv(torch.randn(*s1, dtype=torch.float64), dt) if kind == "ttt" else None
    if kind == "tst":
        fused = lambda: ttnn.addalpha(b, a, scale)
        chain = lambda: ttnn.add(ttnn.multiply(a, scale), b)
    else:
        fused = lambda: ttnn.addcmul(a, c, b, value=1.0)
        chain = lambda: ttnn.add(a, ttnn.multiply(c, b))
    fo, co = fused(), chain()
    f = ttnn.to_torch(fo).to(torch.float64)
    ch = ttnn.to_torch(co).to(torch.float64)
    diff = float((f - ch).abs().max())
    scale_max = float(ch.abs().max()) or 1.0
    ttnn.deallocate(fo); ttnn.deallocate(co)
    for _ in range(WARMUP):
        one(fused); one(chain)
    tf = tc = 0.0
    for i in range(REPS):
        if i % 2 == 0:
            tf += one(fused); tc += one(chain)
        else:
            tc += one(chain); tf += one(fused)
    tf /= REPS; tc /= REPS
    ttnn.deallocate(a); ttnn.deallocate(b)
    if c is not None:
        ttnn.deallocate(c)
    saved = (tc - tf) * 1e6
    rows.append(dict(site=label, calls=calls, kind=kind, dtype=str(dt).split(".")[-1],
                     bit_exact=(diff == 0.0), diff=diff, rel_diff=diff / scale_max,
                     fused_us=tf * 1e6, chain_us=tc * 1e6, saved_us=saved,
                     ratio=tc / tf, total_ms=saved * calls / 1000.0))
    print(f"  {label:26s} x{calls:5d} {str(dt).split('.')[-1]:8s} "
          f"bit-exact={str(diff == 0.0):5s} diff={diff:.3e} | "
          f"{tf*1e6:7.1f} vs {tc*1e6:7.1f} us  {saved*calls/1000.0:8.2f} ms/fold ({tc/tf:.3f}x)",
          flush=True)

print()
for model, pfx, fold_s in (("protenix-v2", "px2", 6.92), ("openfold3", "of3", 9.93)):
    sel = [r for r in rows if r["site"].startswith(pfx)]
    for k, kind in (("scale_add", "tst"), ("mask_add", "ttt")):
        g = [r for r in sel if r["kind"] == kind]
        if not g:
            continue
        tot = sum(r["total_ms"] for r in g)
        nbe = [r["site"] for r in g if not r["bit_exact"]]
        print(f"{model} {k:10s} {tot:7.2f} ms = {100*tot/1000/fold_s:.3f}%  "
              f"non-bit-exact sites: {nbe or 'NONE'}", flush=True)

out = os.environ.get("OUT", "/tmp/exact2.json")
with open(out, "w") as fh:
    json.dump(rows, fh, indent=1)
print("wrote " + out)

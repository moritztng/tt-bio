"""Which ttnn call inside the nlp_create_qkv_heads VJP carries its time.

prof_n128.json puts 84 % of one Evoformer block backward in the 12 `_v_create_qkv_heads`
nodes. This replays one block backward with every ttnn call made from taped_ttnn and
autograd timed and synced, keyed by (function, input shape, call site). Measurement only.
"""
import collections
import sys
import time

import torch

sys.path.insert(0, ".")
from perf.bcx_afgrad import afgrad as A  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
card = int(sys.argv[2]) if len(sys.argv) > 2 else 3
stack = sys.argv[3] if len(sys.argv) > 3 else "evo"
torch.manual_seed(0)
dm, ref = A.load_models(A.DEFAULT_PARAMS)
dev = A.Dev(dm)
tt, ttnn, ag = dev.tt, dev.ttnn, dev.ag
m0, z0 = A.embed(ref["bf16"], torch.randn(n, 20), torch.arange(n))
m0, z0 = m0.detach(), z0.detach()
wm, wz = torch.randn(m0.shape), torch.randn(z0.shape)

per = collections.defaultdict(lambda: [0, 0.0])
TIMED = {"permute", "reshape", "zeros", "concat", "slice", "add", "pad", "typecast",
         "multiply", "matmul", "sum", "transpose"}


def _shape(x):
    if hasattr(x, "shape"):
        return tuple(int(d) for d in x.shape)
    if isinstance(x, (list, tuple)) and x and hasattr(x[0], "shape"):
        return tuple(tuple(int(d) for d in y.shape) for y in x)
    return x


def timed(name, f):
    def g(*a, **k):
        dev.sync()
        fr = sys._getframe(1)
        where = f"{fr.f_code.co_filename.rsplit('/', 1)[-1]}:{fr.f_lineno}"
        t0 = time.time()
        r = f(*a, **k)
        dev.sync()
        e = per[(name, str(_shape(a[0]) if a else k), where)]
        e[0] += 1
        e[1] += time.time() - t0
        return r
    return g


def step(on):
    ml, zl = dev.leaf(m0), dev.leaf(z0)
    with tt.tape():
        mo, zo = dev.stack(ml, zl, *((1, 0) if stack == "extra" else (0, 1)))
    roots = [zo] if stack == "extra" else [mo, zo]
    seeds = [dev.seed(wz, zo)] if stack == "extra" else [dev.seed(wm, mo), dev.seed(wz, zo)]
    dev.sync()
    # the tape is closed here, so the module holds the real verbs; wrap them for the backward
    real = {k: getattr(ttnn, k) for k in TIMED if hasattr(ttnn, k)}
    if on:
        for k, f in real.items():
            setattr(ttnn, k, timed(k, f))
    try:
        t0 = time.time()
        ag.backward(roots, seeds)
        dev.sync()
    finally:
        for k, f in real.items():
            setattr(ttnn, k, f)
    return time.time() - t0


for _ in range(3):
    step(False)
per.clear()
wall = step(True)
rows = sorted(((k, c, s) for k, (c, s) in per.items()), key=lambda r: -r[2])
out = {"stamp": A.stamp(card), "n": n, "stack": stack, "synced_bwd_wall": wall,
       "timed_total": sum(s for _, _, s in rows),
       "calls": [{"fn": k[0], "shape": k[1], "site": k[2], "count": c, "seconds": s}
                 for k, c, s in rows]}
for r in out["calls"][:20]:
    print(r)
print("wall", wall, "timed_total", out["timed_total"])
A.save(f"qkvdiag_n{n}_{stack}.json", out)

"""Per-op synced wall time inside one block backward, bf16 arm against +b8, same process.

A sync after every ttnn call serialises the queue, so each op's wall is its device time plus a
fixed dispatch cost. Absolute numbers are inflated; the arm-to-arm difference per op is what
this reads. Grouped by op name and the dtypes of its tensor operands."""
import sys, time, collections, json
sys.path.insert(0, "perf/bcx_stack")
import stack as S
from types import SimpleNamespace
import ttnn
from ttnn import decorators as D
lv, dev, ref = S.open_all(SimpleNamespace(params=S.A.DEFAULT_PARAMS, card=3))
m0, z0, wm, wz = S.inputs(ref, 224, 0)
import os
dev.ag.DEVICE_ZEROS = bool(int(os.environ.get("BFP8_DEVZ", "0")))
REC = None
orig = D.FastOperation.__call__
def dt(a):
    return str(a.dtype).split(".")[-1].lower().replace("bfloat", "bf").replace("float32", "f32") if isinstance(a, ttnn.Tensor) else None
def timed(self, *a, **k):
    if REC is None:
        return orig(self, *a, **k)
    ttnn.synchronize_device(dev.device)
    t = time.perf_counter()
    out = orig(self, *a, **k)
    ttnn.synchronize_device(dev.device)
    key = self.python_fully_qualified_name.replace("ttnn.", "") + "[" + ",".join(
        x for x in map(dt, list(a) + list(k.values())) if x) + "->" + (dt(out) or "") + "]"
    REC[key][0] += 1
    REC[key][1] += time.perf_counter() - t
    return out
D.FastOperation.__call__ = timed
res = {}
real_bwd = dev.ag.backward
for arm in ("stack", "stack+b8", "stack", "stack+b8"):
    lv.arm(arm)
    REC = None
    S.block_step(dev, lv, m0, z0, wm, wz, "evo", k=1)
    rec = collections.defaultdict(lambda: [0, 0.0])
    def bwd(*a, **k):
        global REC
        REC = rec
        try:
            return real_bwd(*a, **k)
        finally:
            REC = None
    dev.ag.backward = bwd
    S.block_step(dev, lv, m0, z0, wm, wz, "evo", k=1)
    dev.ag.backward = real_bwd
    res[arm] = rec
for arm in ("stack", "stack+b8"):
    r = res[arm]
    print("===", arm, "total synced", round(sum(v[1] for v in r.values()), 4), "calls", sum(v[0] for v in r.values()))
    for k, v in sorted(r.items(), key=lambda kv: -kv[1][1])[:22]:
        print(f"  {v[1]*1e3:8.2f} ms {v[0]:5d}  {k}")
json.dump({a: dict(r) for a, r in res.items()}, open("perf/bcx_bfp8/opsync_n224_devz" + os.environ.get("BFP8_DEVZ", "0") + ".json", "w"), indent=1)

"""Per-op synced wall in one whole block step (taped forward + backward), bf16 against +b8.

A sync after every ttnn call serialises the queue, so absolute walls are inflated by a fixed
dispatch cost per call; the arm-to-arm difference per (phase, op, dtypes) is what this reads.
DEVICE_ZEROS is on in both arms: without it the +b8 backward packs zeros on the host."""
import sys, time, collections, json, os
sys.path.insert(0, "perf/bcx_stack")
import stack as S
from types import SimpleNamespace
import ttnn
from ttnn import decorators as D
N = int(os.environ.get("BFP8_N", "224"))
STACK = os.environ.get("BFP8_STACK", "evo")
lv, dev, ref = S.open_all(SimpleNamespace(params=S.A.DEFAULT_PARAMS, card=3, device_zeros=True))
m0, z0, wm, wz = S.inputs(ref, N, 0)
REC = None
orig = D.FastOperation.__call__
def dt(a):
    if not isinstance(a, ttnn.Tensor):
        return None
    return str(a.dtype).split(".")[-1].lower().replace("bfloat", "bf").replace("float32", "f32")
def timed(self, *a, **k):
    if REC is None:
        return orig(self, *a, **k)
    ttnn.synchronize_device(dev.device)
    t = time.perf_counter()
    out = orig(self, *a, **k)
    ttnn.synchronize_device(dev.device)
    key = (lv.phase + " " + self.python_fully_qualified_name.replace("ttnn.", "") + "[" + ",".join(
        x for x in map(dt, list(a) + list(k.values())) if x) + "->" + (dt(out) or "") + "]")
    REC[key][0] += 1
    REC[key][1] += time.perf_counter() - t
    return out
D.FastOperation.__call__ = timed
res = {}
for arm in ("stack", "stack+b8", "stack", "stack+b8"):
    lv.arm(arm)
    REC = None
    S.block_step(dev, lv, m0, z0, wm, wz, STACK, k=1)
    REC = collections.defaultdict(lambda: [0, 0.0])
    S.block_step(dev, lv, m0, z0, wm, wz, STACK, k=1)
    res[arm], REC = REC, None
for arm in ("stack", "stack+b8"):
    r = res[arm]
    for ph in ("fwd", "bwd"):
        rr = {k: v for k, v in r.items() if k.startswith(ph)}
        print("===", arm, ph, "synced", round(sum(v[1] for v in rr.values()) * 1e3, 2), "ms, calls",
              sum(v[0] for v in rr.values()))
        for k, v in sorted(rr.items(), key=lambda kv: -kv[1][1])[:14]:
            print(f"  {v[1]*1e3:8.2f} ms {v[0]:5d}  {k}")
json.dump({a: dict(r) for a, r in res.items()},
          open(f"perf/bcx_bfp8/opsync_step_{STACK}_n{N}.json", "w"), indent=1)

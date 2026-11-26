"""Run trunk_step with ttnn.from_torch timed and bucketed by (out dtype, layout, torch dtype). Profiling only."""
import sys, time, atexit, collections, runpy
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
import ttnn, torch
orig = ttnn.from_torch
B = collections.defaultdict(lambda: [0, 0.0, 0])
def timed(x, *a, **k):
    t0 = time.perf_counter(); r = orig(x, *a, **k); dt = time.perf_counter() - t0
    key = (str(k.get("dtype", a[0] if a else None)), str(k.get("layout")), str(getattr(x, "dtype", type(x))), "dev" if k.get("device") is not None else "host")
    b = B[key]; b[0] += 1; b[1] += dt; b[2] += getattr(x, "numel", lambda: 0)()
    return r
ttnn.from_torch = timed
@atexit.register
def dump():
    for k, (n, t, e) in sorted(B.items(), key=lambda kv: -kv[1][1]):
        print("FT", k, n, "calls %.3f s %.1f Melem %.1f Melem/s" % (t, e / 1e6, e / 1e6 / max(t, 1e-9)), flush=True)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")

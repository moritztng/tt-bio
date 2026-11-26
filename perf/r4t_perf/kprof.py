"""Per-verb drained attribution of trunk_step K-candidate grad-off trunk pass (batch K). Profiling only.
The 2nd batch-K grad-off Trunk call runs with every ttnn verb in tt-bio modules synchronized inside its timer
(blockprof.W drain), so self time = drained device time + one dispatch; the 1st and 3rd run undrained for the wall.
Writes $KPROF_OUT (json): wall per call, drained sum, top verbs priced against max(flops/115.7T, bytes/440G)."""
import sys, os, time, json, runpy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
import ttnn
import blockprof as BP
from tt_bio.rfd4 import trunk as TR
from tt_bio import autograd as ag, tenstorrent as T
DEV, CALLS, OUT = [], [], {}
_gd = T.get_device
def gd(*a, **k):
    d = _gd(*a, **k)
    DEV[:] = DEV or [d]
    return d
T.get_device = gd
_call = TR.Trunk.__call__
def call(self, s, *a, **k):
    b = int(getattr(getattr(s, "value", s), "shape")[0])
    if ag.is_grad_enabled() or b == 1:
        return _call(self, s, *a, **k)
    i = len(CALLS); drain = i == 1
    ttnn.synchronize_device(DEV[0])
    if drain:
        BP._SYNC = ttnn.synchronize_device; BP.W.drain_dev = DEV[0]; BP.swap(True)
    t0 = time.perf_counter()
    try:
        r = _call(self, s, *a, **k); ttnn.synchronize_device(DEV[0])
    finally:
        if drain:
            BP.swap(False); BP.W.drain_dev = None
    CALLS.append(dict(batch=b, wall_s=round(time.perf_counter() - t0, 4), drained=drain))
    if drain:
        rows = []
        for (name, shp), (n, sns, nb, fl) in BP.REC.verb.items():
            roof = max(fl / BP.COMPUTE_ROOF, nb / BP.DRAM_ROOF)
            rows.append(dict(verb=name, shape=list(shp), n=n, self_s=sns / 1e9, roof_s=roof))
        rows.sort(key=lambda r: -r["self_s"])
        byverb = {}
        for r in rows:
            v = byverb.setdefault(r["verb"], [0, 0.0, 0.0]); v[0] += r["n"]; v[1] += r["self_s"]; v[2] += r["roof_s"]
        OUT.update(drained_sum_s=sum(r["self_s"] for r in rows), roof_sum_s=sum(r["roof_s"] for r in rows),
                   by_verb=sorted(([k, *v] for k, v in byverb.items()), key=lambda x: -x[2])[:40], top=rows[:60])
    OUT["calls"] = CALLS
    json.dump(OUT, open(os.environ["KPROF_OUT"], "w"), indent=1)
    return r
TR.Trunk.__call__ = call
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")

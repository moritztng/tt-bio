"""Split Folds.backward into readback vs host torch fold VJP, inside a real trunk_step run (args passed through)."""
import runpy, sys, time, json, torch
import tt_bio.train.folds as F
from torch.profiler import profile, ProfilerActivity
_rb = F._read_back
T = {"rb": 0.0, "ag": 0.0, "n": 0, "elems": 0}
def rb(g, s):
    t = time.perf_counter(); r = _rb(g, s); T["rb"] += time.perf_counter() - t; T["n"] += 1; T["elems"] += r.numel(); return r
F._read_back = rb
_ab = torch.autograd.backward
step = [0]
def ab(*a, **k):
    step[0] += 1
    t = time.perf_counter()
    if step[0] == 3:
        with profile(activities=[ProfilerActivity.CPU]) as p:
            r = _ab(*a, **k)
        print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=12), flush=True)
    else:
        r = _ab(*a, **k)
    T["ag"] += time.perf_counter() - t
    print("FOLDPROF", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in T.items()}), flush=True)
    for k in T: T[k] = 0 if k in ("n", "elems") else 0.0
    return r
F.torch.autograd.backward = ab
sys.argv = ["trunk_step.py"] + sys.argv[1:]
runpy.run_path("perf/r4t_step/trunk_step.py", run_name="__main__")

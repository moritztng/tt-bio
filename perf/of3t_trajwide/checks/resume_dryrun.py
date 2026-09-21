"""Exercise every line of resume.load_ours against the real 4.1 GB checkpoint the live arm
just wrote, with a fake optimizer and fake params, so a format or signature error cannot
first show up as an arm that dies at startup and loops on the supervisor."""
import sys, types, numpy as np
sys.path.insert(0, "/home/ttuser/.coworker/wt/of3t-trajwide/perf/of3t_trajwide")
import resume

D = "/home/ttuser/of3t_runs/trajwide/w/shipped"

class FakeDev:
    pass

class FakeVal:
    def __init__(s, a): s.a = a; s.dtype = "float32"
    def device(s): return FakeDev()

class FakeT:
    def __init__(s, a): s.value = FakeVal(a)

# what the real call site passes
with np.load(D + "/resume.npz") as z:
    names = sorted({k.split("//", 1)[1] for k in z.files if k != "__meta__"
                    and k.startswith("master//")})
print("checkpoint carries", len(names), "master tensors")

opt = types.SimpleNamespace(master={}, exp_avg={}, exp_avg_sq={}, steps=-1)
params = {n: FakeT(None) for n in names}
params_obj = dict(params)

class Params(dict):
    def rebind(s):
        s.rebound = True
        return len(s)

P = Params(params)
pushed = {}
def to_device(arr, dev, dtype=None):
    pushed[id(arr)] = (arr.shape, dtype)
    return FakeVal(arr)

# the dump k01.npz is in checkpoint orientation; the live orient() reproduces it. Here the
# assertion is exercised by handing back exactly the dump, which is what a correct restore does.
with np.load(D + "/k01.npz") as z:
    dump = {n: z[n] for n in z.files}
log = []
k, stale, fwd = resume.load_ours(D, opt, P, to_device, log, lambda: dump)
print("k =", k, "| steps =", opt.steps, "| stale =", stale is not None, "| fwd_rel =", fwd)
print("master", len(opt.master), "exp_avg", len(opt.exp_avg), "exp_avg_sq", len(opt.exp_avg_sq))
print("pushed to device:", len(pushed), "| rebind called:", getattr(P, "rebound", False))
print("log rows restored:", len(log), log[-1] if log else None)
assert k == 1 and len(opt.master) == len(names) and len(pushed) == len(names)
assert opt.steps >= 1 and len(log) == 1
print("DRY RUN OK")

# and the negative control: a restore that lands on different weights must NOT pass
bad = {n: v.copy() for n, v in dump.items()}
n0 = sorted(bad)[0]
bad[n0] = bad[n0] + np.float32(1e-7)
opt2 = types.SimpleNamespace(master={}, exp_avg={}, exp_avg_sq={}, steps=-1)
try:
    resume.load_ours(D, opt2, Params(params), to_device, [], lambda: bad)
    print("NEGATIVE CONTROL FAILED: a perturbed restore passed")
except AssertionError as e:
    print("negative control OK:", str(e).splitlines()[0][:130])
import os
os.replace(D + "/resume.npz.BAD", D + "/resume.npz")   # the guard quarantined it; put it back
print("checkpoint restored for the live arm:", os.path.exists(D + "/resume.npz"))

"""Host memory traffic of a trunk_step step, per phase: every aten op on CPU tensors, its wall time and the bytes of its
CPU inputs + outputs, keyed by (phase, op, largest shape). Steps before --from are skipped (warm-up). numpy work (the host
optimizer, grad .numpy()) is not seen. Profiling only: run as `hostbytes.py <from_step> trunk_step.py ... --pcie-meter on`."""
import sys, time, json, atexit, collections, runpy, os
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pcie_meter as PM

FROM = int(sys.argv[1]); sys.argv = sys.argv[2:]
S = {"phase": "pre", "step": 0}
A = collections.defaultdict(lambda: [0, 0.0, 0])   # (phase, op, shape) -> [calls, s, bytes]
_phase = PM.phase
def phase(name):
    if name == "fwd":
        S["step"] += 1
    S["phase"] = name
    return _phase(name)
PM.phase = phase

def nb(ts):
    out, big = 0, ()
    for t in ts:
        if isinstance(t, torch.Tensor) and t.device.type == "cpu":
            n = t.numel() * t.element_size(); out += n
            if n and (not big or n > big[0]): big = (n, tuple(t.shape), str(t.dtype).replace("torch.", ""))
    return out, big

class M(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        t0 = time.perf_counter(); r = func(*args, **(kwargs or {})); dt = time.perf_counter() - t0
        if S["step"] >= FROM:
            bi, b1 = nb(tree_flatten((args, kwargs))[0]); bo, b2 = nb(tree_flatten(r)[0])
            big = max((b1, b2), key=lambda b: b[0] if b else 0)
            k = (S["phase"], str(func.overloadpacket.__name__), str(big[1:]) if big else "")
            a = A[k]; a[0] += 1; a[1] += dt; a[2] += bi + bo
        return r

@atexit.register
def dump():
    n = max(1, S["step"] - FROM + 1)
    rows = sorted(((p, o, s, c / n, t / n, b / n) for (p, o, s), (c, t, b) in A.items()), key=lambda x: -x[5])
    ph = collections.defaultdict(lambda: [0.0, 0.0])
    for p, o, s, c, t, b in rows: ph[p][0] += t; ph[p][1] += b
    print("HOSTBYTES_STEPS", n, json.dumps({p: dict(s=round(t, 3), GB=round(b / 1e9, 3)) for p, (t, b) in ph.items()}), flush=True)
    for r in rows[:60]:
        print("HB %-10s %-28s %-44s calls %6.1f  %7.4f s  %8.1f MB" % (r[0], r[1], r[2][:44], r[3], r[4], r[5] / 1e6), flush=True)

with M():
    runpy.run_path(sys.argv[0], run_name="__main__")

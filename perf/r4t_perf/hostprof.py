"""Wall time of the host-side pieces of a trunk_step step (glue, heads, loss, bridge/fold backward, optimizer). Profiling only."""
import sys, time, atexit, collections, runpy, json
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
from tt_bio.rfd4 import network as N
from tt_bio.train import bridge as BR, folds as FO, optim as OP, rfd4_objective as RO
B = collections.defaultdict(list)
def wrap(owner, name, label=None):
    f = getattr(owner, name)
    def w(*a, **k):
        t0 = time.perf_counter(); r = f(*a, **k); B[label or name].append(time.perf_counter() - t0); return r
    setattr(owner, name, w)
G = N.RFD4Network if hasattr(N, "RFD4Network") else None
cls = [getattr(N, c) for c in dir(N) if isinstance(getattr(N, c), type) and hasattr(getattr(N, c), "initial_states")][0]
for m in ("initial_states", "conditioning", "heads"):
    wrap(cls, m)
wrap(BR.Bridge, "backward", "bridge_backward")
wrap(FO.Folds, "tape_backward")
wrap(OP.AdamW, "step", "opt_step")
wrap(RO, "rfd4_loss")
@atexit.register
def dump():
    print("HOSTPROF", json.dumps({k: [round(x, 4) for x in v] for k, v in B.items()}), flush=True)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")

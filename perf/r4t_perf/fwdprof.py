"""Split a trunk_step forward into its pieces (host glue, K-candidate card trunk, heads/loss, winner glue, taped winner trunk).
Each wrapped call synchronizes the device after it, so card time lands on the piece that issued it. Profiling only."""
import sys, time, atexit, collections, runpy, json
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
import ttnn
from tt_bio.rfd4 import network as N, trunk as TR
from tt_bio.rfd4 import pair as PR
from tt_bio.train import rfd4_objective as RO, bridge as BR
from tt_bio import autograd as ag
B = collections.defaultdict(list)
DEV = []
def sync():
    if DEV: ttnn.synchronize_device(DEV[0])
def wrap(owner, name, label):
    f = getattr(owner, name)
    def w(*a, **k):
        sync(); t0 = time.perf_counter(); r = f(*a, **k); sync()
        lab = label + ("_taped" if label == "trunk" and ag.is_grad_enabled() else "")
        B[lab].append(time.perf_counter() - t0); return r
    setattr(owner, name, w)
_od = ttnn.open_device if hasattr(ttnn, "open_device") else None
from tt_bio import tenstorrent as T
_gd = T.get_device
def gd(*a, **k):
    d = _gd(*a, **k)
    if not DEV: DEV.append(d)
    return d
T.get_device = gd
wrap(N.FlowGlue, "initial_states", "glue_init")
wrap(N.FlowGlue, "heads", "heads")
wrap(PR.PairSplit, "constant", "pair_constant")
wrap(RO, "rfd4_loss", "loss")
tr = [getattr(TR, c) for c in dir(TR) if isinstance(getattr(TR, c), type) and c == "Trunk"]
wrap(tr[0], "__call__", "trunk")
@atexit.register
def dump():
    print("FWDPROF", json.dumps({k: [round(x, 4) for x in v] for k, v in B.items()}), flush=True)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")

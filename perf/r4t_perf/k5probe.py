"""K5 probe: why the trimul gated move is never asked in the grad-off K pass. Logs, per TriangleMultiplication call,
mask shape, taping, NOGRAD_IS_INFERENCE, fast/raw flags, and every eligible_gated / F1-tail answer. Wraps ksplit.py."""
import sys, os, json, runpy, collections
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
from tt_bio import tenstorrent as T, ops, reblock_permute as RP
LOG = collections.Counter()
_c = T.TriangleMultiplication.__call__
def call(self, x, *a, **k):
    m = k.get("mask", a[0] if a else None)
    LOG[json.dumps(dict(x=[int(s) for s in getattr(x, "shape", [])], mask=None if m is None else [int(s) for s in m.shape],
        taping=bool(ops.taping()), nogi=ops.NOGRAD_IS_INFERENCE, fast=T._FAST_MODE, raw=T._TRIMUL_RAW_CHANNEL_MOVES,
        gm=self.gated_move, mam=T._TRIMUL_MASK_AFTER_MOVE))] += 1
    return _c(self, x, *a, **k)
T.TriangleMultiplication.__call__ = call
_e = RP.eligible_gated
def eg(*a, **k):
    r = _e(*a, **k); LOG["eligible_gated=" + str(r) + " " + str([int(s) for s in a[0].shape])] += 1; return r
RP.eligible_gated = eg
import atexit
def dump():
    open(os.environ["K5P_OUT"], "w").write(json.dumps(LOG, indent=1))
_exit = os._exit
def ex(c): dump(); _exit(c)
os._exit = ex
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")

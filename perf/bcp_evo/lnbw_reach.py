#!/usr/bin/env python3
"""Stage 13 reach probe: of the lnbw dx contributions on real Evoformer blocks, how many land on a
tensor that already holds a gradient (the fusable case), and how many are first and get a later add.
Wraps perf/bcp_evo/vjp_grade.py (every stage on). perf/bcp_evo/lnbw_reach.py [vjp_grade args]"""
import atexit, collections, pathlib, runpy, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import autograd as AG, lnbw
C, marks, firsts = collections.Counter(), set(), set()
_bw, _add = lnbw.layer_norm_bw, AG.Tensor.add_grad
def bw(*a, **k):
    r = _bw(*a, **k); marks.add(id(r)); return r
def add_grad(self, grad):
    if id(grad) in marks:
        marks.discard(id(grad)); C["lnbw dx"] += 1
        if self._grad is not None or self._pend is not None: C["onto an accumulator"] += 1
        elif self._parts or self._slab is not None: C["onto pending slices"] += 1
        else: C["first contribution"] += 1; firsts.add(id(self))
    elif id(self) in firsts:
        firsts.discard(id(self)); C["first, then another add"] += 1
    return _add(self, grad)
lnbw.layer_norm_bw, AG.Tensor.add_grad = bw, add_grad
atexit.register(lambda: print("LNBW_REACH", dict(C), flush=True))
sys.argv = ["vjp_grade.py"] + sys.argv[1:]
runpy.run_path(str(ROOT / "perf/bcp_evo/vjp_grade.py"), run_name="__main__")

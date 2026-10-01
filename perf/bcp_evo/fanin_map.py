#!/usr/bin/env python3
"""Which VJP rules produce the contributions that meet at a fan-in add, on real Evoformer blocks.
Wraps perf/bcp_evo/vjp_grade.py (every stage on). perf/bcp_evo/fanin_map.py [vjp_grade args]"""
import atexit, collections, pathlib, runpy, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import autograd as AG
SEEN, PAIRS, SHAPES = collections.defaultdict(list), collections.Counter(), {}
_add, _slice = AG.Tensor.add_grad, AG.Tensor.add_grad_slice
def producer():
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if "tt_bio" in fn and not (fn.endswith("autograd.py") and f.f_code.co_name in (
                "add_grad", "add_grad_slice", "_join_parts", "_join_slab", "grad", "closure_grad")):
            return f"{pathlib.Path(fn).stem}.{f.f_code.co_name}:{f.f_lineno}"
        f = f.f_back
    return "?"
def add_grad(self, grad):
    if self.requires_grad:
        SEEN[id(self)].append(producer())
        SHAPES[id(self)] = (tuple(int(d) for d in self.value.shape), str(grad.dtype))
    return _add(self, grad)
AG.Tensor.add_grad = add_grad
def report():
    for k, v in SEEN.items():
        if len(v) > 1:
            PAIRS[(SHAPES[k][0], " + ".join(v))] += 1
    print("FANIN_MAP", flush=True)
    for (shape, pr), n in PAIRS.most_common():
        print(f"  {n:3d}  {shape}  {pr}", flush=True)
atexit.register(report)
sys.argv = ["vjp_grade.py"] + sys.argv[1:]
runpy.run_path(str(ROOT / "perf/bcp_evo/vjp_grade.py"), run_name="__main__")

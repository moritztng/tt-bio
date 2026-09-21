#!/usr/bin/env python3
"""Two of this census's classes are read off the source; this reads them off the card.

`READABLE_MASS.json` files 0.27153 % of the gradient mass as FUSED_NOT_SPLIT and 0.10450 % as
NOT_A_LEAF_LAZY. Both classifications came from reading `openfold3_diffusion_transformer.py`
and `openfold3_atom_transformer.py` and checking that the tensor counts land exactly. That is
good evidence and it is not a measurement, and the two claims are different in kind:

  FUSED_NOT_SPLIT says the fused `qkv_w` / `qkv_b` IS a registered leaf and DOES receive a
  gradient, and only the inverse selection is missing. If it turned out to receive none, the
  class would be wrong and the remedy would be a port change rather than an instrument one.

  NOT_A_LEAF_LAZY says the atom transformer's other weights do not exist yet when the walk that
  registers leaves runs, because `_w_tt` builds them inside `forward`. If they were already
  present, the class would be wrong and something else would be blocking them.

Both are decided here by running the SHIPPED instrument unmodified and looking at the module it
built. `of3t-diffusion`'s `device_gradient.py` is another row's and is not touched: this wraps
`of3_coverage._device_weights`, which `device_gradient.py` imports by name inside `main()`, so
the wrapper keeps a handle on the module and on the pre-forward walk. Afterwards the same
module is walked again, and every walked tensor is asked whether the tape gave its leaf a
gradient. The diff between the two walks is the lazy set, measured rather than argued.

No timing is claimed anywhere here, so no clock is quoted: this is a presence census.
"""
from __future__ import annotations

import json
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf", "of3t_tape"))
sys.path.insert(0, os.path.join(os.getcwd(), "perf", "of3t_gradients"))

import of3_coverage                                                    # noqa: E402
import tt_bio.autograd as ag                                           # noqa: E402

_real_walk = of3_coverage._device_weights
_seen = {}


def _spy(obj, prefix="", seen=None, out=None, depth=0):
    w = _real_walk(obj, prefix, seen, out, depth)
    if depth == 0 and prefix == "" and "module" not in _seen:
        _seen["module"] = obj
        _seen["before"] = dict(w)
    return w


of3_coverage._device_weights = _spy

script = "perf/of3t_diffusion/device_gradient.py"
sys.argv = [script] + sys.argv[1:]
rc = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    rc = e.code or 0

out = {"instrument": "of3t-readable-mass leafprobe.py -- the shipped diffusion gradient "
                     "instrument run unmodified, with the module's own weight walk read "
                     "before and after the forward",
       "wrapped": script, "argv": sys.argv[1:], "instrument_exit": rc}

if "module" not in _seen:
    out["error"] = "the walk was never called; the instrument did not reach the module build"
else:
    before, mod = _seen["before"], _seen["module"]
    after = _real_walk(mod)
    lazy = sorted(set(after) - set(before))
    import ttnn                                                        # noqa: E402

    def grad_of(t):
        leaf = ag._PARAMS.get(id(t))
        if leaf is None:
            return None, "not a registered leaf"
        g = getattr(leaf, "grad", None)
        if g is None:
            return None, "leaf, no gradient"
        gv = g.value if hasattr(g, "value") else g
        return float(ttnn.to_torch(gv).double().norm()), "leaf with a gradient"

    rows = {}
    for path, t in sorted(before.items()):
        n, why = grad_of(t)
        rows[path] = {"grad_norm": n, "state": why}
    n_leaf_grad = sum(1 for r in rows.values() if r["grad_norm"] is not None)

    qkv = {p: r for p, r in rows.items() if p.endswith(("qkv_w", "qkv_b"))}
    out["walk"] = {
        "reachable_before_the_forward": len(before),
        "reachable_after_the_forward": len(after),
        "materialised_inside_the_forward": len(lazy),
        "materialised_names_sample": lazy[:40],
        "note": "only the tensors in the BEFORE walk are registered as leaves: "
                "device_gradient.py calls ag.parameter over that walk and never again. A "
                "tensor that first exists during the forward can therefore never receive a "
                "gradient on this step, whatever the tape does with it.",
    }
    out["leaves"] = {
        "walked_before": len(before),
        "with_a_gradient": n_leaf_grad,
        "without_a_gradient": len(before) - n_leaf_grad,
    }
    out["fused_qkv"] = {
        "n": len(qkv),
        "with_a_gradient": sum(1 for r in qkv.values() if r["grad_norm"] is not None),
        "rows": qkv,
        "claim_under_test": "FUSED_NOT_SPLIT asserts these leaves receive a gradient and that "
                            "only the inverse selection onto their linear_q/k/v is missing.",
    }
    lazy_rows = {}
    for path in lazy:
        n, why = grad_of(after[path])
        lazy_rows[path] = {"grad_norm": n, "state": why}
    out["lazy_atom_transformer"] = {
        "n_materialised_inside_the_forward": len(lazy),
        "of_those_registered_as_a_leaf":
            sum(1 for r in lazy_rows.values() if r["state"] != "not a registered leaf"),
        "of_those_with_a_gradient":
            sum(1 for r in lazy_rows.values() if r["grad_norm"] is not None),
        "rows": lazy_rows,
        "claim_under_test": "NOT_A_LEAF_LAZY asserts the atom transformer's non-AdaLN weights "
                            "are absent from the pre-forward walk, are therefore never passed "
                            "to ag.parameter, and receive no gradient. Both halves are read "
                            "here: the walk diff, and the tape's own answer for each of them.",
    }

dst = os.path.join("perf", "of3t_readable_mass", "LEAFPROBE.json")
os.makedirs(os.path.dirname(dst), exist_ok=True)
json.dump(out, open(dst, "w"), indent=2, default=str)
print(json.dumps({k: v for k, v in out.items() if k not in ("fused_qkv", "lazy_atom_transformer")},
                 indent=2, default=str)[:4000], flush=True)
print("wrote", dst, flush=True)
sys.exit(rc)

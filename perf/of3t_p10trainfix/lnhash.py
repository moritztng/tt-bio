#!/usr/bin/env python3
"""trainarm.py for ONE step with every layer-norm backward hashed: g, x, dx, dgamma, dbeta.

`gradhash.py` put the entry of the run-to-run divergence between a transition's weight
gradients (identical) and the gradient its layer norm hands back to the residual stream
(different). Two runs of this, diffed call by call in tape order, name the first call whose
inputs agree and whose outputs do not.

    lnhash.py --hash-out <json> <trainarm argv...>
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
i = sys.argv.index("--hash-out")
OUT = Path(sys.argv[i + 1])
del sys.argv[i:i + 2]

import numpy as np                                                    # noqa: E402
from tt_bio import autograd as ag                                     # noqa: E402
from tt_bio.train.tensors import to_host                              # noqa: E402

_orig = ag._layer_norm_bw
_add = ag.Tensor.add_grad
calls = []
CAP = {}


def add_grad(self, grad):
    c = CAP.get(id(self))
    if c is not None:
        c[1][c[0]] = _h(grad)
    return _add(self, grad)


ag.Tensor.add_grad = add_grad


def _h(v):
    if v is None:
        return None
    a = np.ascontiguousarray(to_host(v).astype(np.float32))
    return [hashlib.sha1(a.tobytes()).hexdigest()[:12], float(np.linalg.norm(a))]


def _layer_norm_bw(x, gamma, beta, eps, bwcfg):
    bw = _orig(x, gamma, beta, eps, bwcfg)

    def wrapped(g):
        rec = {"i": len(calls), "shape": [int(d) for d in x.value.shape],
               "dtype": str(x.value.dtype), "g": _h(g), "x": _h(x.value),
               "gamma": _h(gamma.value) if gamma is not None else None}
        got = {}
        CAP.clear()
        for name, t in (("dx", x), ("dgamma", gamma), ("dbeta", beta)):
            if t is not None:
                CAP[id(t)] = (name, got)
        try:
            bw(g)
        finally:
            CAP.clear()
        rec.update(got)
        calls.append(rec)
        if len(calls) % 50 == 0:
            OUT.write_text(json.dumps(calls) + "\n")
    return wrapped


ag._layer_norm_bw = _layer_norm_bw
import atexit                                                         # noqa: E402
atexit.register(lambda: OUT.write_text(json.dumps(calls) + "\n"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "of3t_p10trainout"))
import trainarm                                                       # noqa: E402

raise SystemExit(trainarm.main())

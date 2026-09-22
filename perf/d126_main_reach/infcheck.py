#!/usr/bin/env python3
"""Does the slot -> property change reach INFERENCE?

The change is confined to `autograd.Tensor`. So the question is whether an untaped forward
constructs one, and that is a count rather than an argument. This runs the shipped
`tt_bio.ops.linear` route -- the same verb the tape hooks -- with no tape installed, counts
every `autograd.Tensor` construction and every `value` read, and hashes the output.

Scope: this is the ops surface the change could touch, not a whole protein model. A fold
exercises more ops; it exercises no more of THIS change, because nothing outside
`autograd.Tensor` was edited.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np                                                  # noqa: E402
import tt_bio.autograd as ag                                        # noqa: E402
from tt_bio import ops                                              # noqa: E402
from perf.train_d_dp import model as M                              # noqa: E402

built = {"n": 0}
_real_init = ag.Tensor.__init__


def counting_init(self, value, requires_grad=False):
    built["n"] += 1
    return _real_init(self, value, requires_grad)


ag.Tensor.__init__ = counting_init

assert ops.grad_hook() is None, "a tape is installed; this is not the inference path"
assert not ag.installed()

ds = M.Dataset(1, 512, 256, seed=0)
trunk = M.Trunk(512, 256, 2, seed=0)
out = trunk(ds.batch([0]))
from tt_bio.train.tensors import to_host                            # noqa: E402
arr = np.asarray(to_host(out["pred_xyz"]), dtype=np.float32)

rec = {
    "value_is_property": isinstance(getattr(ag.Tensor, "value", None), property),
    "has_parameter_for": hasattr(ag, "parameter_for"),
    "grad_hook_installed": ag.installed(),
    "autograd_tensors_constructed_during_inference": built["n"],
    "out_shape": list(arr.shape),
    "out_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    "out_sum": float(arr.sum()),
}
print(json.dumps(rec, indent=2))

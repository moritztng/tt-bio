"""What the old `Tensor.free` did to the input of a taped move: the trimul tail's reallocate-then-
deallocate sequence, on whatever tree is first on PYTHONPATH.

Records whether the freed handle came back allocated (and in which buffer), whether its bytes are
the reallocated output's, and whether the gradient through the move is still exact.
"""
import json
import sys

import torch
import ttnn

from tt_bio import autograd as ag
from tt_bio import taped_ttnn as tt
from tt_bio import tenstorrent as T

dev = T.get_device()
mc = ttnn.L1_MEMORY_CONFIG
x = ag.Tensor(ttnn.from_torch(torch.randn(1, 1, 64, 64), dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc),
              requires_grad=True)
with tt.tape():
    a = tt.taped_ttnn().typecast(x, ttnn.float32, memory_config=mc)
    b = tt.taped_ttnn().reallocate(a)
    tt.taped_ttnn().deallocate(a)
r = {"tree": ag.__file__, "freed_input_allocated_after_free": a.value.is_allocated()}
if a.value.is_allocated():
    r["its_buffer"] = str(a.value.memory_config().buffer_type)
    r["its_bytes_equal_the_moved_output"] = bool(torch.equal(ttnn.to_torch(a.value),
                                                             ttnn.to_torch(b.value)))
ag.backward([b], [None])
r["grad_exact"] = bool(torch.equal(ttnn.to_torch(x.grad).float(), torch.ones(1, 1, 64, 64)))
print("RESULT", json.dumps(r), flush=True)

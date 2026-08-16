#!/usr/bin/env python3
"""Is the Transition fusion F1s kernel with a different activation, or a third kernel?

F1 computes `p * sigmoid(g)` for `p = xa @ wa`, `g = xb @ wb`. Transitions swiglu computes
`silu(x_norm @ fc1) * (x_norm @ fc2)`: the same two-GEMM-one-eltwise shape, with silu instead of
sigmoid, applied to the other operand, and ONE shared activation instead of two. This asks
`trimul_tail.eligible()` which of its clauses Transitions real 512 aa chunk shape trips, so the
continuation knows whether the build is an extension or a rewrite.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio import trimul_tail as TT

dev = T.get_device()
def dram(t):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
print("BLOCK =", TT.BLOCK)
for name, xs, ws in (
        ("transition chunk, L1 operands", (1, 16, 512, 256), (256, 1024)),
        ("transition fc3",                (1, 16, 512, 1024), (1024, 256)),
        ("trimul tail 512 (control)",     (1, 512, 512, 256), (256, 256))):
    x = dram(torch.randn(*xs, dtype=torch.bfloat16))
    w = dram(torch.randn(*ws, dtype=torch.bfloat16))
    mt = 1
    for d in [int(d) for d in x.padded_shape][:-1]:
        mt *= d
    mt = (mt + 31) // 32
    print(f"{name:32s} in={xs} w={ws} mt={mt} kt={ws[0]//32} nt={ws[1]//32} -> "
          + str(TT.eligible(x, x, w, w) or "ELIGIBLE"))
    ttnn.deallocate(x); ttnn.deallocate(w)

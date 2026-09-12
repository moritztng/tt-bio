"""Does the cached generic_op descriptor follow BOTH operands when their addresses move?

The layer changes `in0` every call and `in1` only when a weight is replaced. The block-level
negative control for the fused gate came back dead, so this asks the narrow question: with the
descriptor already cached, does a new weight buffer at a new address reach the kernel?
"""
import torch
import ttnn

from tt_bio import tenstorrent as tt
from tt_bio import mm_dualnoc as DN

S = 512
dev = tt.get_device()
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
f = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
MC = ttnn.DRAM_MEMORY_CONFIG
xt = torch.randn(1, S, S, 128)
wt = torch.cat([torch.randn(128, 512), torch.randn(128, 128)], dim=-1) * 0.05


def call(w, fresh_x):
    """One fused call, with in0 freshly allocated so its address moves like the layer's does."""
    x = f(xt) if fresh_x else X0
    outs = DN.in_proj(x, w, KC, ttnn.bfloat16, MC, split=(512, 128))
    g = ttnn.to_torch(outs[1])
    a = (x.buffer_address(), w.buffer_address())
    for o in outs:
        ttnn.deallocate(o)
    if fresh_x:
        ttnn.deallocate(x)
    return g, a


X0 = f(xt)
W1 = f(wt)
g1, a1 = call(W1, True)
g1b, a1b = call(W1, True)
print(f"same weight, fresh in0 twice: in0 {a1[0]}->{a1b[0]}  equal={torch.equal(g1, g1b)}")

bad = wt.clone()
bad[:, -1] += 0.5
W2 = f(bad)
g2, a2 = call(W2, True)
print(f"new weight buffer: in1 {a1[1]} -> {a2[1]}   gate moved={not torch.equal(g1, g2)}"
      f"  max|d|={(g1.float()-g2.float()).abs().max().item():.6g}")

# and once more with the ORIGINAL weight object, to show the descriptor follows it back
g3, a3 = call(W1, True)
print(f"back to the first weight: in1 {a3[1]}  equal to first result={torch.equal(g1, g3)}")

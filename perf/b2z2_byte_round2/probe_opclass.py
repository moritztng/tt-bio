"""Does a `ttnn.linear` column of a pair projection equal the same column of a `minimal_matmul`?

Both remaining redundant reads in the block are owned by a `ttnn.linear` sitting next to a
`minimal_matmul` over the same activation (trace rows 22/51 and 58/86 of
`perf/b2z2_byte_floor/out/trace_512_wh_c10.json.gz`). Folding one into the other deletes the read,
but it also moves those columns from one matmul kernel to the other, and `_trimul_out_proj`'s own
docstring says the two "block the contraction differently, so bf16 accumulates in a different
order". If that is true at these shapes the whole row cannot be bit-exact and has to say so before
anything is built.

It is answerable without building anything: run the two ops at the two shapes and compare. The
second question is whether a matmul's N width changes its own arithmetic -- it must not, since each
output tile is an independent contraction -- and that is checked by slicing a wide result.

Usage: TT_VISIBLE_DEVICES=<n> TT_BIO_LEASE_CARDS=<n> python3 probe_opclass.py
"""
import os
import sys

import torch
import ttnn

from tt_bio import tenstorrent as tt

S = int(os.environ.get("PROBE_TOKENS", "512"))
dev = tt.get_device()
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)

f = lambda t, shape=None: ttnn.from_torch(
    t if shape is None else t.reshape(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    device=dev)
th = lambda t: ttnn.to_torch(t).float()

ok = True


def cmp(name, a, b):
    global ok
    ta, tb = th(a), th(b)
    eq = torch.equal(ta, tb)
    d = float((ta - tb).abs().max())
    ok = ok and eq
    print(f"  {name:<46} torch.equal {str(eq):<5} max abs {d:.6g}", flush=True)
    return eq


def lin(x, w):
    return ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                       compute_kernel_config=KC, program_config=tt._pair_proj_config(x, w))


def mm(x, w):
    return ttnn.experimental.minimal_matmul(
        x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
        compute_kernel_config=KC)


print(f"=== R2: the trimul gate, [1,{S},{S},128] x [128,128] ===", flush=True)
x4 = f(torch.randn(1, S, S, 128))
w_in = f(torch.randn(128, 512))          # the fused four-way in-projection
w_g = f(torch.randn(128, 128))           # g_out
a, b = lin(x4, w_g), mm(x4, w_g)
cmp("ttnn.linear(g_out) vs minimal_matmul", a, b)
# the same columns as the tail of a 640-wide matmul: N width must not change the arithmetic
w_cat = f(torch.cat([th(w_in), th(w_g)], dim=-1).bfloat16().float())
wide = mm(x4, w_cat)
cmp("wide[512:640] vs narrow g_out (mm)", ttnn.slice(wide, [0, 0, 0, 512], [1, S, S, 640]), b)
cmp("wide[0:512] vs narrow in-proj (mm)", ttnn.slice(wide, [0, 0, 0, 0], [1, S, S, 512]),
    mm(x4, w_in))
ttnn.deallocate(wide)
ttnn.deallocate(x4)

print(f"=== R1b: the tri-attention bias, [{S},{S},128] x [128,32] ===", flush=True)
x3 = f(torch.randn(S, S, 128))
w_b = f(torch.randn(128, 32))
w_qkvg = f(torch.randn(128, 512))
cmp("ttnn.linear(bias) vs minimal_matmul", lin(x3, w_b), mm(x3, w_b))
w_cat2 = f(torch.cat([th(w_qkvg), th(w_b)], dim=-1).bfloat16().float())
wide2 = mm(x3, w_cat2)
cmp("wide[512:544] vs narrow bias (mm)", ttnn.slice(wide2, [0, 0, 512], [S, S, 544]), mm(x3, w_b))

print("VERDICT:", "bit-exact everywhere" if ok else "NOT bit-exact -- see the failing line")
sys.exit(0 if ok else 3)

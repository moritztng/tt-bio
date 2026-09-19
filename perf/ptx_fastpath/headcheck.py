#!/usr/bin/env python3
"""Gradients of the head-packing verbs, against a float64 reference.

Same evidence order as `perf/hallgrad/gradcheck.py`: the float64 reference is validated
against float64 central finite differences before the device is compared to it. These two
verbs are pure rearrangement, so the failure they hide is not a magnitude error -- it is
one head's gradient landing on another head, which looks entirely healthy.

The loss is a WEIGHTED sum over a chain that uses all three outputs with different
weights, so q, k and v gradients are distinguishable and a slot swap cannot pass.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, ttnn
from hallgrad.gradcheck import COS_BAR, REL_L2_BAR, fd_check, metrics
from tt_bio import tenstorrent as tt
from tt_bio import autograd as ag

B, L, H, dh = 1, 64, 4, 32
rng = np.random.default_rng(0)
x64 = torch.tensor(rng.standard_normal((B, 1, L, 3 * H * dh)), dtype=torch.float64)
wq = torch.tensor(rng.standard_normal((B, H, L, dh)), dtype=torch.float64)
wk = torch.tensor(rng.standard_normal((B, H, L, dh)), dtype=torch.float64) * 2.0
wv = torch.tensor(rng.standard_normal((B, H, L, dh)), dtype=torch.float64) * 3.0
wo = torch.tensor(rng.standard_normal((B, 1, L, H * dh)), dtype=torch.float64) * 5.0
bf = lambda t: t.to(torch.bfloat16).to(torch.float64)


def ref(x):
    s = x.reshape(B, 1, L, 3, H, dh)
    q, k, v = (s[:, 0, :, i].permute(0, 2, 1, 3) for i in range(3))
    o = q.permute(0, 2, 1, 3).reshape(B, 1, L, H * dh)
    return (q * wq).sum() + (k * wk).sum() + (v * wv).sum() + (o * wo).sum()


p = x64.clone().requires_grad_(True)
worst, probed, elig = fd_check(lambda: ref(p), [p], n_probe=40, seed=0)
print(f"reference vs float64 central differences: worst {worst:.2e} "
      f"over {probed} of {elig} eligible")
assert worst < 1e-6, "reference failed its own check"

r = bf(x64).requires_grad_(True)
ref(r).backward()

dev = tt.get_device()
D = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=dev)
xa = ag.Tensor(D(x64), requires_grad=True)
# `tape()` rebinds `ttnn` inside tt-bio's modules, and this file is not one of them, so
# the taped surface is addressed directly. A shipped module calls it as plain `ttnn.`.
tt_ttnn = ag._SHIM
with ag.tape():
    q, k, v = tt_ttnn.experimental.nlp_create_qkv_heads(
        xa, num_heads=H, num_kv_heads=H, transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    o = tt_ttnn.experimental.nlp_concat_heads(q, memory_config=ttnn.DRAM_MEMORY_CONFIG)
# forward, against the reference split
for nm, t, c in (("q", q, 0), ("k", k, 1), ("v", v, 2)):
    c_ref = bf(x64).reshape(B, 1, L, 3, H, dh)[:, 0, :, c].permute(0, 2, 1, 3)
    same = torch.equal(ttnn.to_torch(t.value).to(torch.float64), c_ref)
    print(f"forward {nm}: {'BIT-EXACT' if same else 'DIFFERS'}")
ag.backward([q, k, v, o], [D(wq), D(wk), D(wv), D(wo)])
m = metrics(ttnn.to_torch(xa.grad).to(torch.float64).numpy(), r.grad.detach().numpy())
ok = m["rel_l2"] <= REL_L2_BAR and m["cos"] >= COS_BAR
print(f"dx  rel_l2 {m['rel_l2']:.3e}  cos {m['cos']:.6f}  max_abs {m['max_abs']:.3e}  "
      f"{'PASS' if ok else 'FAIL'}")

# Negative control: the check must be able to fail. Swap two slots in the reference and
# confirm the comparison rejects it -- otherwise a slot swap would have passed silently.
bad = r.grad.detach().clone().reshape(B, 1, L, 3, H, dh)
bad[:, :, :, [0, 1]] = bad[:, :, :, [1, 0]]
mb = metrics(ttnn.to_torch(xa.grad).to(torch.float64).numpy(),
             bad.reshape(B, 1, L, 3 * H * dh).numpy())
print(f"negative control (q and k slots swapped): rel_l2 {mb['rel_l2']:.3e} "
      f"{'-- rejected, the check can fail' if mb['rel_l2'] > REL_L2_BAR else '-- CHECK IS BLIND'}")
ok = ok and mb["rel_l2"] > REL_L2_BAR
print("VERDICT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

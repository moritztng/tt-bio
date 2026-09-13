"""Does folding `g_out` into the trimul's in-projection change a number? torch.equal or bust.

One process, one card, one built PairformerLayer, the same activation through both arms. The lever
moves the trimul's output gate from a `ttnn.linear` into a second destination of the
in-projection's `minimal_matmul`, so the comparison is on the whole block's output: an op-class
change that were not bit-exact would show up in the residual.

THE FIXTURE HAS TO BE UNZEROED FIRST. `tt_bio.reference` zero-initialises every sub-unit's output
projection so the residual starts as identity, and with `p_out.weight` zero the trimul contributes
exactly nothing -- `p_out * sigmoid(g_out)` is zero whatever the gate is. A block-level comparison
against that fixture passes for both arms and for a deliberately WRONG arm, which is how this check
first read BIT-EXACT with a negative control that could not fail. Every all-zero weight is given a
small random value below, and the negative control then fires.
"""
import sys

import torch
import ttnn

from tt_bio import tenstorrent as tt
from tt_bio import reference as ref
from tt_bio import triatt_qkv as K
from tt_bio import reblock_permute as RP

S = int(sys.argv[1]) if len(sys.argv) > 1 else 512
dev = tt.get_device()
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
rl = ref.PairformerLayer(384, 128, 16, 0.25, 32, 4, v2=True)
weights = {k: v.float() for k, v in rl.state_dict().items()}
zeroed = [k for k, v in weights.items() if v.numel() and float(v.abs().max()) == 0.0]
for k in zeroed:
    weights[k] = torch.randn_like(weights[k]) * 0.05
print(f"unzeroed {len(zeroed)} weights so every sub-unit reaches the output: {zeroed}")
layer = tt.PairformerLayer(32, 4, 24, 16, True, weights, KC)

s = torch.randn(1, S, 384)
z = torch.randn(1, S, S, 128)
m1 = torch.ones(1, S)
f = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
mask_tt = f(m1[:, :, None] * m1[:, None, :])
attn_tt = f((1 - m1).unsqueeze(1).unsqueeze(1) * -1e9)


def census():
    return {"trimul_gout": list(tt.TRIMUL_GOUT_STATS), "qkvg": list(K.QKVG_STATS),
            "qkv_heads": list(K.STATS), "tail": list(K.TAIL_STATS),
            "trimul_tail_f1": list(tt._trimul_tail.STATS),
            "reblock_gated": int(RP.STATS_GATED[0])}


def run(on):
    tt.set_trimul_fused_gout(on)
    c0 = census()
    out_s, out_z = layer(f(s), f(z), mask_tt, attn_tt, attn_tt)
    r = (ttnn.to_torch(out_s), ttnn.to_torch(out_z))
    ttnn.deallocate(out_s)
    ttnn.deallocate(out_z)
    c1 = census()
    return r, {k: [b - a for a, b in zip(c0[k], c1[k])] if isinstance(c1[k], list)
               else c1[k] - c0[k] for k in c1}


run(False)                                  # compile both arms before anything is compared
run(True)

base, cb = run(False)
lev, cl = run(True)
print(f"census base  {cb}")
print(f"census lever {cl}")
if cl["trimul_gout"][0] == 0:
    print("THE LEVER NEVER SERVED — nothing was compared."
          f"  rejects={dict(tt.TRIMUL_GOUT_REJECTS)}")
    raise SystemExit(2)

ok = True
for name, a, b in (("s", base[0], lev[0]), ("z", base[1], lev[1])):
    same = torch.equal(a, b)
    ok &= same
    print(f"  {name:<3} torch.equal={same}  max|d|="
          f"{(a.float() - b.float()).abs().max().item():.6g}  shape={tuple(a.shape)}")

# negative control: one column of the gate half of every fused weight, and the lever again
for tm in (layer.triangle_multiplication_start, layer.triangle_multiplication_end):
    for key, w in list(tm._gp_gout_cache.items()):
        wt = ttnn.to_torch(w).float()
        wt[:, -1] += 0.5
        tm._gp_gout_cache[key] = f(wt)
bad, _ = run(True)
neg = not torch.equal(bad[1], lev[1])
print(f"negative control (perturbed fused gate column) changes the block output: {neg}"
      f"  max|d|={(bad[1].float() - lev[1].float()).abs().max().item():.6g}")
print("VERDICT:", "BIT-EXACT" if (ok and neg) else "FAILED")
raise SystemExit(0 if (ok and neg) else 1)

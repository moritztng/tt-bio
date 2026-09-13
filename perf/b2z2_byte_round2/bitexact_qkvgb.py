"""Does folding the pair-bias projection into the qkv+gate pass change a number? torch.equal or bust.

The comparison is against the arm with `qkvg` already on, because that is what this lever is
marginal to: it adds a fifth, one-tile-wide output chunk to a matmul that already writes four.
Both arms run in one process on one card on the same activation.

The fixture is UNZEROED -- `tt_bio.reference` zero-initialises 23 of this layer's weights,
including both attentions' `linear_o` and `linear_g`, and a block-output comparison against that
is blind to whole sub-units (see `FINDINGS.md`).
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
print(f"unzeroed {len(zeroed)} weights")
layer = tt.PairformerLayer(32, 4, 24, 16, True, weights, KC)

s = torch.randn(1, S, 384)
z = torch.randn(1, S, S, 128)
m1 = torch.ones(1, S)
f = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
mask_tt = f(m1[:, :, None] * m1[:, None, :])
attn_tt = f((1 - m1).unsqueeze(1).unsqueeze(1) * -1e9)

K._QKVG_ENABLED = True                      # the arm this lever is marginal to, on in both arms


def census():
    return {"qkvgb": list(K.QKVGB_STATS), "qkvg": list(K.QKVG_STATS),
            "qkv_heads": list(K.STATS), "tail": list(K.TAIL_STATS),
            "trimul_tail_f1": list(tt._trimul_tail.STATS),
            "reblock_gated": int(RP.STATS_GATED[0])}


def run(on):
    K._QKVGB_ENABLED = bool(on)
    c0 = census()
    out_s, out_z = layer(f(s), f(z), mask_tt, attn_tt, attn_tt)
    r = (ttnn.to_torch(out_s), ttnn.to_torch(out_z))
    ttnn.deallocate(out_s)
    ttnn.deallocate(out_z)
    c1 = census()
    return r, {k: [b - a for a, b in zip(c0[k], c1[k])] if isinstance(c1[k], list)
               else c1[k] - c0[k] for k in c1}


run(False)
run(True)

base, cb = run(False)
lev, cl = run(True)
print(f"census base  {cb}")
print(f"census lever {cl}")
if cl["qkvgb"][0] == 0:
    print(f"THE LEVER NEVER SERVED — nothing was compared. rejects={dict(K.QKVGB_REJECTS)}")
    raise SystemExit(2)

ok = True
for name, a, b in (("s", base[0], lev[0]), ("z", base[1], lev[1])):
    same = torch.equal(a, b)
    ok &= same
    print(f"  {name:<3} torch.equal={same}  max|d|="
          f"{(a.float() - b.float()).abs().max().item():.6g}  shape={tuple(a.shape)}")

# negative control, in the BIAS columns specifically: the last tile of the fused weight is the
# bias, and perturbing it must move the block. Perturbing it is the only way to prove the fused
# call's fifth chunk is what the attention is reading.
for ta in (layer.triangle_attention_start, layer.triangle_attention_end):
    w = ttnn.to_torch(ta.qkvgb_weight).float()
    w[:, -32:] += 0.5
    ta.qkvgb_weight = f(w)
bad, _ = run(True)
neg = not torch.equal(bad[1], lev[1])
print(f"negative control (perturbed bias columns of the fused weight) moves the block: {neg}"
      f"  max|d|={(bad[1].float() - lev[1].float()).abs().max().item():.6g}")
print("VERDICT:", "BIT-EXACT" if (ok and neg) else "FAILED")
raise SystemExit(0 if (ok and neg) else 1)

"""Is the fused qkv+gate projection the same numbers as the two it replaces? torch.equal or bust.

Runs both arms in one process on one card, on the same activation and the same weights, and
compares q, k, v and the gate element by element. A negative control (one weight column perturbed)
has to make the comparison FAIL, else the comparison is not reading what it claims to.

Also reports the eligibility census, because a fused call that silently declined would compare two
copies of the same path and pass for the wrong reason.
"""
import os
import sys

import torch
import ttnn

from tt_bio import tenstorrent as tt
from tt_bio import triatt_qkv as K

S = int(sys.argv[1]) if len(sys.argv) > 1 else 512
dev = tt.get_device()
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)

n_heads, head_dim, c_z = 4, 32, 128
c = n_heads * head_dim
torch.manual_seed(0)
w_qkv = torch.randn(3 * c, c_z) * 0.05
w_g = torch.randn(c, c_z) * 0.05
w_o = torch.randn(c_z, c) * 0.05
x = torch.randn(1, S, S, c_z)

f = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
x_tt = ttnn.reshape(f(x), (S, S, c_z))
W_QKV, W_G, W_O = f(w_qkv.t()), f(w_g.t()), f(w_o.t())


def arms(w_g_torch):
    """(separate, fused) as torch tensors, for the given gate weight."""
    wg = f(w_g_torch.t())
    q, k, v = K.qkv_heads(x_tt, W_QKV, KC, n_heads, head_dim, ttnn.bfloat16,
                          tt._qkv_mm_config(x_tt, W_QKV))
    g = K.gate_proj(x_tt, wg, W_O, KC, n_heads, head_dim, ttnn.bfloat16,
                    tt._qkv_mm_config(x_tt, wg))
    sep = [ttnn.to_torch(t) for t in (q, k, v, g)]
    W = f(torch.cat([w_qkv, w_g_torch], dim=0).t())
    fus = K.qkvg_heads(x_tt, W, W_O, KC, n_heads, head_dim, ttnn.bfloat16,
                       tt._qkv_mm_config(x_tt, W))
    if fus is None:
        return sep, None
    (fq, fk, fv), fg = fus
    return sep, [ttnn.to_torch(t) for t in (fq, fk, fv, fg)]


os.environ["TT_BIO_TRIATT_FUSED_QKVG"] = "1"
K._QKVG_ENABLED = True
before = list(K.QKVG_STATS)
sep, fus = arms(w_g)
print(f"eligibility: qkvg served {K.QKVG_STATS[0] - before[0]}, refused "
      f"{K.QKVG_STATS[1] - before[1]}  rejects={dict(K.QKVG_REJECTS)}")
print(f"             qkv_heads {K.STATS}  gate/tail {K.TAIL_STATS}")
if fus is None:
    print("FUSED PATH DECLINED — nothing was compared"); raise SystemExit(2)

names = ("q", "k", "v", "gate")
ok = True
for n, a, b in zip(names, sep, fus):
    same = torch.equal(a, b)
    ok &= same
    print(f"  {n:<5} torch.equal={same}  max|d|={(a.float()-b.float()).abs().max().item():.6g}"
          f"  shape={tuple(a.shape)}")

# negative control: the comparison must be able to fail
w_bad = w_g.clone()
w_bad[0, 0] += 0.5
sep2, fus2 = arms(w_bad)
neg = not torch.equal(sep2[3], fus[3])
print(f"negative control (perturbed gate column) differs from the unperturbed fused gate: {neg}")
print("VERDICT:", "BIT-EXACT" if (ok and neg) else "FAILED")
raise SystemExit(0 if (ok and neg) else 1)

"""The split writer, on its own: two destinations of one matmul against the two matmuls it replaces.

`mm_dualnoc.in_proj(..., split=(a, b))` must produce exactly what `in_proj` produces for each half
separately, and a perturbed weight column must move the half it belongs to and only that half.
"""
import sys

import torch
import ttnn

from tt_bio import tenstorrent as tt
from tt_bio import mm_dualnoc as DN

S = int(sys.argv[1]) if len(sys.argv) > 1 else 512
dev = tt.get_device()
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
f = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
MC = ttnn.DRAM_MEMORY_CONFIG

x = f(torch.randn(1, S, S, 128))
wa_t = torch.randn(128, 512) * 0.05
wg_t = torch.randn(128, 128) * 0.05


def halves(wa, wg):
    w = f(torch.cat([wa, wg], dim=-1))
    outs = DN.in_proj(x, w, KC, ttnn.bfloat16, MC, split=(512, 128))
    if outs is None:
        print("SPLIT REFUSED:", dict(DN.REJECTS)); raise SystemExit(2)
    r = [ttnn.to_torch(o) for o in outs]
    for o in outs:
        ttnn.deallocate(o)
    return r


ref_a = ttnn.to_torch(DN.in_proj(x, f(wa_t), KC, ttnn.bfloat16, MC))
ref_g = ttnn.to_torch(DN.in_proj(x, f(wg_t), KC, ttnn.bfloat16, MC))
sa, sg = halves(wa_t, wg_t)
ok = True
for n, a, b in (("in-proj half", ref_a, sa), ("gate half", ref_g, sg)):
    same = torch.equal(a, b)
    ok &= same
    print(f"  {n:<14} torch.equal={same}  max|d|={(a.float()-b.float()).abs().max().item():.6g}"
          f"  shape={tuple(b.shape)}")

print("column sensitivity — a perturbed weight column must move its own half and nothing else:")
neg_ok = True
for label, dwa, dwg in (("in-proj col 0", 0, None), ("gate col 0", None, 0),
                        ("gate col 127", None, 127)):
    a2, b2 = wa_t.clone(), wg_t.clone()
    if dwa is not None:
        a2[:, dwa] += 0.5
    else:
        b2[:, dwg] += 0.5
    pa, pg = halves(a2, b2)
    da = (pa.float() - sa.float()).abs().max().item()
    dg = (pg.float() - sg.float()).abs().max().item()
    want_a, want_g = (dwa is not None), (dwg is not None)
    good = (da > 0) == want_a and (dg > 0) == want_g
    neg_ok &= good
    print(f"  {label:<14} d(in-proj)={da:.6g}  d(gate)={dg:.6g}   {'ok' if good else 'WRONG'}")

print("VERDICT:", "SPLIT CORRECT" if (ok and neg_ok) else "FAILED")
raise SystemExit(0 if (ok and neg_ok) else 1)

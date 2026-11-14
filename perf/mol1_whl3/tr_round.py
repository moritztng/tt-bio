"""Trunk block 43's single Transition (upstream fp32 weights + captured inputs): which bf16 rounding inside it
reproduces WH's small-channel token error? python -I tr_round.py CKPT SPLIT.pt UP_LAYERS.pt"""
import sys, torch
import torch.nn.functional as F
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
sd = ck.get("model", ck.get("state_dict", ck))
keys = [k for k in sd if "single_transition" in k and ".blocks.43." in k and "confidence" not in k and "pairformer_stack" in k]
print(keys)
g = lambda suf: sd[[k for k in keys if k.endswith(suf)][0]].double()
lw, lb, wa, wb, wo = g("layernorm1.weight"), g("layernorm1.bias"), g("linear_no_bias_a.weight"), g("linear_no_bias_b.weight"), g("linear_no_bias.weight")
r = torch.load(sys.argv[2], weights_only=False); up = torch.load(sys.argv[3], weights_only=False)
small = up[-1].abs().amax(0) < 512
x = r["tr.in"][0].double().reshape(-1, 384); ref = r["tr.out"].double().reshape(-1, 384)
bf = lambda t: t.to(torch.bfloat16).double()
b8 = lambda t: t  # placeholder
def run(rx=lambda t: t, rln=lambda t: t, rab=lambda t: t, rh=lambda t: t, rw=lambda t: t):
    xn = rln(F.layer_norm(rx(x), (384,), rw(lw), rw(lb), 1e-5))
    a, b = rab(xn @ rw(wa).T), rab(xn @ rw(wb).T)
    h = rh(F.silu(a) * b)
    return h @ rw(wo).T
def tok(t): return t - t.mean(0)
def rel(a, b): return float((a - b).norm() / b.norm())
o64 = run(); print("fp64 recompute vs captured fp32 out, small tok rel", rel(tok(o64[:, small]), tok(ref[:, small])))
for name, kw in (("bf16 input s", dict(rx=bf)), ("bf16 LN out", dict(rln=bf)), ("bf16 a,b", dict(rab=bf)),
                 ("bf16 hidden", dict(rh=bf)), ("bf16 weights", dict(rw=bf)),
                 ("bf16 LN+hidden", dict(rln=bf, rh=bf)), ("all bf16", dict(rx=bf, rln=bf, rab=bf, rh=bf, rw=bf))):
    o = run(**kw)
    print(f"{name:16s} small tok rel {rel(tok(o[:, small]), tok(o64[:, small])):.3g}  tokstd {tok(o[:, small]).std():.3g}")
def bfp8(t, blk=16, mbits=7):
    s = t.reshape(-1, blk) if t.shape[-1] % blk == 0 else None
    e = torch.floor(torch.log2(s.abs().amax(1, keepdim=True).clamp_min(1e-300)))
    step = 2.0 ** (e - (mbits - 1))
    return (torch.round(s / step) * step).reshape(t.shape)
for name, kw in (("bfp8 hidden", dict(rh=bfp8)), ("bfp8 a,b", dict(rab=bfp8)), ("bfp8 weights", dict(rw=lambda t: bfp8(t) if t.dim() == 2 else t)),
                 ("bfp8 LN out", dict(rln=bfp8)), ("bf16 LN, bfp8 hidden+weights", dict(rln=bf, rh=bfp8, rw=lambda t: bfp8(t) if t.dim() == 2 else bf(t)))):
    o = run(**kw)
    print(f"{name:28s} small tok rel {rel(tok(o[:, small]), tok(o64[:, small])):.3g}  tokstd {tok(o[:, small]).std():.3g}")

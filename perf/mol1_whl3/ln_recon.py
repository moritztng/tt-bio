"""8k83 tapfix: WH's fp32_layer_norm intermediates (multiply_ x_hat, multiply_ w*x_hat) vs float64 LN of WH's own input;
then fc1 on each candidate LN output (fp64 / bf16 RNE / bf16 truncated / WH x_hat-based) vs WH's fc1 and upstream.
python -I ln_recon.py CKPT SPLIT.pt TAP.pt UP_LAYERS.pt"""
import sys, torch
import torch.nn.functional as F
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True); sd = sd.get("model", sd)
pre = "module.pairformer_stack.blocks.43.single_transition."
lw, lb, wa, wb = (sd[pre + k].double() for k in ("layernorm1.weight", "layernorm1.bias", "linear_no_bias_a.weight", "linear_no_bias_b.weight"))
r = torch.load(sys.argv[2], weights_only=False); tap = torch.load(sys.argv[3], weights_only=False); up = torch.load(sys.argv[4], weights_only=False)
n = r["tr.in"][0].reshape(-1, 384).shape[0]; small = up[-1].abs().amax(0) < 512
g = lambda t: t.reshape(-1, t.shape[-1]).double()[:n]
x = g(tap["in"][0]); xhat_wh, wx_wh = g(tap["multiply_"][0]), g(tap["multiply_"][1]); fc1_wh = g(tap["_transition_linear:fc1"][0])
tok = lambda t: t - t.mean(0); rel = lambda a, b: float((tok(a) - tok(b)).norm() / tok(b).norm())
mu = x.mean(-1, keepdim=True); xc = x - mu; xhat = xc / (xc.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()
print("x_hat WH vs fp64 (all ch tok rel)", rel(xhat_wh, xhat), " small ch", rel(xhat_wh[:, small], xhat[:, small]))
print("w*x_hat WH vs fp64", rel(wx_wh, xhat * lw))
ln64 = xhat * lw + lb
print("LN out small ch: |mean| max", float(ln64[:, small].mean(0).abs().max()), "tok std", float(tok(ln64[:, small]).std()),
      " bias small", lb[small].tolist()[:8], " w small", lw[small].tolist()[:8])
print("all ch: LN |val| max", float(ln64.abs().max()), "tok std", float(tok(ln64).std()))
def trunc(t):  # fp32 -> bf16 by truncation
    i = t.float().view(torch.int32) & ~0xFFFF; return i.view(torch.float32).double()
rne = lambda t: t.to(torch.bfloat16).double()
U1 = F.silu(r["tr.a.out"].double().reshape(-1, 1536))
for name, ln in (("fp64", ln64), ("bf16 RNE", rne(ln64)), ("bf16 trunc", trunc(ln64)), ("WH fp32 wx + b, RNE", rne(wx_wh + lb)),
                 ("WH fp32 wx + b, trunc", trunc(wx_wh + lb)), ("fp32", ln64.float().double())):
    a = F.silu(ln @ wa.T)
    print(f"{name:22s} fc1 vs upstream {rel(a, U1):.3g}  vs WH fc1 {rel(fc1_wh, a):.3g}")

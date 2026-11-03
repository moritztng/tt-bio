"""WH's LayerNorm in trunk block 43's single Transition vs float64 on WH's own input: per-token scale and shift error.
python -I ln_cmp.py CKPT TAP.pt"""
import sys, torch
import torch.nn.functional as F
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True); sd = sd.get("model", sd)
pre = "module.pairformer_stack.blocks.43.single_transition.layernorm1."
lw, lb = sd[pre + "weight"].double(), sd[pre + "bias"].double()
tap = torch.load(sys.argv[2], weights_only=False)
x = tap["in"][0].double().reshape(-1, 384); wh = tap["layer_norm"][0].double().reshape(-1, 384)
mu, var = x.mean(-1, keepdim=True), x.var(-1, unbiased=False, keepdim=True)
xh = (x - mu) / (var + 1e-5).sqrt()
ref = xh * lw + lb
# WH's normalized value per token: (wh - lb) / lw ~ a_t * xh + c_t; fit a_t, c_t by least squares
y = (wh - lb) / lw
A = torch.stack([xh, torch.ones_like(xh)], -1)
sol = torch.linalg.lstsq(A, y.unsqueeze(-1)).solution.squeeze(-1)
a, c = sol[:, 0], sol[:, 1]
print(f"per-token scale a: mean {a.mean():.6f} std {a.std():.3g} min {a.min():.6f} max {a.max():.6f}")
print(f"per-token shift c: mean {c.mean():.3g} std {c.std():.3g}")
res = y - (a[:, None] * xh + c[:, None])
print(f"residual after per-token affine fit, rms {res.pow(2).mean().sqrt():.3g}; xh rms 1; bf16 step at |y|~1: 7.8e-3")
print(f"x per token: mean |mu| {mu.abs().mean():.3g}, sqrt(var) {var.sqrt().mean():.3g}, rms(x) {x.pow(2).mean(-1).sqrt().mean():.3g}")
d = wh - ref
print(f"LN out error: rms {d.pow(2).mean().sqrt():.3g}; token-part of error rms {(d - d.mean(0)).pow(2).mean().sqrt():.3g}; "
      f"token-part of ref rms {(ref - ref.mean(0)).pow(2).mean().sqrt():.3g}")
bfr = ref.to(torch.bfloat16).double() - ref
print(f"bf16-rounded fp64 LN: error rms {bfr.pow(2).mean().sqrt():.3g}, token-part rms {(bfr - bfr.mean(0)).pow(2).mean().sqrt():.3g}")

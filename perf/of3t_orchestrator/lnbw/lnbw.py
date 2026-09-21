"""Is tt_bio/autograd.py's LayerNorm backward FORMULA correct, independent of the device?

The trunk's residual is backward-only (forward 1.0345x, gradient 5.4139x upstream's own bf16) and
localised to attention_pair_bias and single_transition, whose error mass is 74 % LayerNorm affines.
So the LN backward is a prime suspect. This checks the algebra in float64 against torch autograd,
which separates "the formula is wrong" from "the kernel is imprecise" -- two different fixes.

Transcribed from tt_bio/autograd.py:580-596:
    mean = mean(x, -1); centered = x - mean
    var  = mean(centered^2, -1); rstd = 1/sqrt(var + eps); norm = centered * rstd
    dgamma = sum_leading(g * norm);  dbeta = sum_leading(g)
    dnorm  = g * gamma
    dx     = (dnorm - mean(dnorm) - norm * mean(dnorm*norm)) * rstd
"""
import torch
torch.manual_seed(7)
N, C, EPS = 40, 96, 1e-5

x = torch.randn(3, N, C, dtype=torch.float64, requires_grad=True)
gamma = torch.randn(C, dtype=torch.float64, requires_grad=True)
beta = torch.randn(C, dtype=torch.float64, requires_grad=True)
g = torch.randn(3, N, C, dtype=torch.float64)

ref = torch.nn.functional.layer_norm(x, (C,), weight=gamma, bias=beta, eps=EPS)
ref.backward(g)

with torch.no_grad():
    mean = x.mean(-1, keepdim=True)
    centered = x - mean
    var = (centered * centered).mean(-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + EPS)
    norm = centered * rstd
    dgamma = (g * norm).reshape(-1, C).sum(0)
    dbeta = g.reshape(-1, C).sum(0)
    dnorm = g * gamma
    dn_mean = dnorm.mean(-1, keepdim=True)
    dn_norm_mean = (dnorm * norm).mean(-1, keepdim=True)
    dx = (dnorm - dn_mean - norm * dn_norm_mean) * rstd

for name, ours, theirs in (("dx", dx, x.grad), ("dgamma", dgamma, gamma.grad),
                           ("dbeta", dbeta, beta.grad)):
    rel = float((ours - theirs).norm() / theirs.norm())
    print("%-7s rel %.6e  %s" % (name, rel, "AGREES" if rel < 1e-12 else "DISAGREES"))

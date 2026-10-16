"""LayerNorm folded into trimul T1's matmuls, priced against float64 on dumped real inputs (CPU only).

T1 today: xn = bf16(LN(z) * gamma + beta), then xn @ W (fp32 accumulate). Folded: no xn at all,
LN(z) @ W = rstd * (z @ W' - mu * colsum(W')) + beta @ W with W' = bf16(gamma * W), so the kernel
needs only the per-row mean and rstd. The cancellation in z @ W' - mu * colsum(W') is the risk;
both paths are scored against the same float64 reference, never against each other.
usage: lnfold_err.py t1_*.pt
"""
import sys, torch

def rel(a, ref):
    return ((a.double() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()

for f in sys.argv[1:]:
    d = torch.load(f)
    z = d["z"].reshape(-1, d["z"].shape[-1])                        # bf16, exact input
    g, b = d["gamma"].reshape(-1).double(), d["beta"].reshape(-1).double()
    W = torch.cat([d["wpT"], d["wgT"], d["wo"].t()], 0).double()      # [out, K]
    zd = z.double()
    mu, var = zd.mean(-1, keepdim=True), zd.var(-1, unbiased=False, keepdim=True)
    rstd = (var + 1e-5).rsqrt()
    ref = ((zd - mu) * rstd * g + b) @ W.t()
    Wb = W.to(torch.bfloat16).float()
    # today: LN in fp32, xn rounded to bf16, bf16 weights, fp32 accumulate
    zf = z.float(); muf, rstdf = zf.mean(-1, keepdim=True), (zf.var(-1, unbiased=False, keepdim=True) + 1e-5).rsqrt()
    xn = (((zf - muf) * rstdf) * g.float() + b.float()).to(torch.bfloat16).float()
    cur = xn @ Wb.t()
    # folded: W' = bf16(gamma W), fp32 accumulate on raw z, fp32 correction
    Wp = (W * g).to(torch.bfloat16).float()
    fold = rstdf * (zf @ Wp.t() - muf * Wp.sum(-1)) + (b.float() @ Wb.t())
    ratio = (muf.abs() / rstdf.reciprocal()).squeeze(-1)
    print(f"{f}: call {d['call']} |mu|/std median {ratio.median():.3f} p99 {ratio.quantile(0.99):.3f} max {ratio.max():.3f} | "
          f"rel_rms vs f64: today {rel(cur, ref):.3e}  folded {rel(fold, ref):.3e}")

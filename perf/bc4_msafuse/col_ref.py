"""Float64 reference for AF2's column attention at MSA depth 2 (`AF2Attention._column_depth2`) and its VJP.

`forward` is the function a fused kernel has to compute from the normed MSA, `vjp` the hand-written backward a
kernel would run (recompute, then one pass per MSA row). Running this file checks `vjp` against torch autograd in
float64. Card-free: python3 perf/bc4_msafuse/col_ref.py
"""
import torch

H, D = 8, 32  # heads, head width: one head is one 32-column tile


def forward(xn, wq, wk, wv, wg, bg, wo, bo, gap_bias=None):
    """xn [2, n, c] (LayerNorm output). Weights [c, c] as x @ w. Returns y [2, n, c] and the saved intermediates."""
    q, k, v = xn @ wq, xn @ wk, xn @ wv
    kd, vd = k[0] - k[1], v[0] - v[1]
    n, c = xn.shape[1:]
    gap = D ** -0.5 * (q * kd).reshape(2, n, H, D).sum(-1, keepdim=True)   # logit gap per head
    if gap_bias is not None:
        gap = gap + gap_bias.reshape(1, n, 1, 1)
    p = torch.sigmoid(gap).expand(2, n, H, D).reshape(2, n, c)
    att = v[1] + p * vd
    gate = torch.sigmoid(xn @ wg + bg)
    return (att * gate) @ wo + bo, (q, kd, vd, p, att, gate)


def vjp(dy, saved, wq, wk, wv, wg, wo):
    """dL/dxn from dL/dy. Per MSA row s: three products into the hidden, one head-wise row sum, four back out."""
    q, kd, vd, p, att, gate = saved
    n, c = dy.shape[1:]
    dh = dy @ wo.T
    dxn = (dh * att * gate * (1 - gate)) @ wg.T
    datt = dh * gate
    heads = lambda t: t.reshape(2, n, H, D).sum(-1, keepdim=True).expand(2, n, H, D).reshape(2, n, c)
    dgap = heads(datt * vd) * p * (1 - p) * D ** -0.5
    dvd = (datt * p).sum(0)
    dkd = (dgap * q).sum(0)
    dv = torch.stack([dvd, datt.sum(0) - dvd])
    dk = torch.stack([dkd, -dkd])
    return dxn + (dgap * kd) @ wq.T + dk @ wk.T + dv @ wv.T


if __name__ == "__main__":
    torch.manual_seed(0)
    n, c = 224, H * D
    r = lambda *s: torch.randn(*s, dtype=torch.float64)
    xn = r(2, n, c).requires_grad_()
    wq, wk, wv, wg, wo = (r(c, c) / c ** 0.5 for _ in range(5))
    bg, bo, gb, dy = r(c), r(c), r(1, n, 1), r(2, n, c)
    y, saved = forward(xn, wq, wk, wv, wg, bg, wo, bo, gb)
    (ref,) = torch.autograd.grad(y, xn, dy)
    got = vjp(dy, [t.detach() for t in saved], wq, wk, wv, wg, wo)
    rel = float((got - ref).norm() / ref.norm())
    print(f"vjp rel L2 against autograd (float64): {rel:.3e}")
    assert rel < 1e-12

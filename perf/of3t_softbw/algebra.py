"""The fused route computes the SHIPPED expression: checked in float64, on the host.

`autograd.softmax_bw`'s fused branch is `moreh(y/s, g) * s` with `s = rowsum(y)`, claimed
equal to the composed `y (g - sum(g y)/s)`. The claim is one line of algebra --
`moreh(y/s, g) = (y/s)(g - sum(g y)/s)` -- and one line of algebra is exactly the kind of
thing that is wrong. So it is checked before the device run rather than by it, on a y whose
row sums are deliberately off, which is the only regime where the renorm does anything.

Run: /home/moritz/tt-bio/env/bin/python3 perf/of3t_softbw/algebra.py
"""
import json
import torch


def shipped(y, g):
    inner = (g * y).sum(-1, keepdim=True) / y.sum(-1, keepdim=True)
    return y * (g - inner)


def moreh(y, g):
    """What `ttnn.moreh_softmax_backward(y, g, dim=-1)` computes."""
    return y * (g - (g * y).sum(-1, keepdim=True))


def fused_renorm(y, g):
    """What `autograd.softmax_bw` does on the fused route."""
    s = y.sum(-1, keepdim=True)
    return moreh(y / s, g) * s


def main(seed=0, B=2, H=4, N=64, M=128, rowsum_spread=0.04):
    torch.manual_seed(seed)
    x = torch.randn(B, H, N, M, dtype=torch.float64) * 8
    # a y whose row sums are off, the way `ttnn.softmax` returns one: up to 3.88e-02 by
    # measurement (perf/of3t_f64softmax/ROWSUM_PROBE.json)
    y = torch.softmax(x, -1) * (1 + rowsum_spread * torch.randn(B, H, N, 1, dtype=torch.float64))
    g = torch.randn(B, H, N, M, dtype=torch.float64)
    a, b, m = shipped(y, g), fused_renorm(y, g), moreh(y, g)
    rep = {
        "row_sum_max_abs_dev_from_one": float((y.sum(-1) - 1).abs().max()),
        "fused_renorm_vs_shipped_rel_l2": float((a - b).norm() / a.norm()),
        "dropping_the_renorm_vs_shipped_rel_l2": float((m - a).norm() / a.norm()),
        "d_logits_row_sum_max_abs": {
            "shipped": float(a.sum(-1).abs().max()),
            "fused_renorm": float(b.sum(-1).abs().max()),
            "plain_moreh": float(m.sum(-1).abs().max()),
        },
    }
    print(json.dumps(rep, indent=1))
    assert rep["fused_renorm_vs_shipped_rel_l2"] < 1e-14, "the fused route is NOT the shipped one"
    return rep


if __name__ == "__main__":
    main()

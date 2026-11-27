"""Training-side taped wrapper for RFD4 PairOnCard (r4t-perf P2).

Lives under `tt_bio/train/` so inference module `tt_bio/rfd4/pair.py` never imports `autograd`.
"""
from __future__ import annotations

import torch
import ttnn

from .. import autograd as ag
from ..rfd4.pair import PairInputs, PairOnCard


def pair_taped(pc: PairOnCard, p: PairInputs, bridge, batch: dict, dtype=None):
    """The training winner's z_init: the program `pc.__call__` runs, taped. Its backward stays on the card
    and hands the builder's projection and norm weights and the AdaLN's gamma and beta their gradients, so
    neither the pair nor its gradient crosses PCIe and the host never runs a pair-sized backward.

        n1 = x r1,  a = n1 w,  b = a r2,  z = b gamma + beta
        dgamma = sum g b,  dbeta = sum g,  da = r2 (g gamma - b mean(g gamma b)),  dw = sum da n1
        dx = r1 (da w - n1 mean(da w n1)),  dW = dx^T F
    """
    b = int(p.bins.shape[0])
    assert b == 1, "one taped winner a step"
    z = pc(p, 0)
    if dtype is not None and z.dtype != dtype:
        z32, z = z, ttnn.typecast(z, dtype)
        ttnn.deallocate(z32)
    fac = pc.split.builder.init_repr_factory
    W, lnw = fac.linear_out.weight, fac.ln_out.weight
    n, c = int(p.bins.shape[1]), int(W.shape[0])
    leaves = [bridge.up(t, ttnn.float32) for t in (W, lnw.view(1, c), p.gamma.reshape(1, c), p.beta.reshape(1, c))]
    if not any(t.requires_grad for t in leaves):
        return ag.Tensor(z)
    e1, e2 = float(fac.ln_out.eps), float(pc.split.builder.adaln.norm.eps)
    feats = pc.split.features(batch).reshape(n * n, -1)
    k = int(feats.shape[-1])
    kp = -(-k // 32) * 32
    F = ttnn.from_torch(torch.nn.functional.pad(feats, [0, kp - k]).contiguous(), dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT, device=pc.device)
    const = pc._const[2]
    bins = p.bins
    row = lambda v: ttnn.from_torch(v.detach().float().reshape(1, 1, 1, c), dtype=ttnn.float32,
                                    layout=ttnn.TILE_LAYOUT, device=pc.device)
    w_tt, g_tt = row(lnw), row(p.gamma)

    def make():
        def bw(g):
            cfg = ag.precise_config()
            mean = lambda v: ag._row_mean(v, cfg, c)
            if g.dtype != ttnn.float32:
                g = ttnn.typecast(g, ttnn.float32)
            x = pc._x(const, bins)
            r1 = ttnn.rsqrt(ttnn.add(mean(ttnn.multiply(x, x)), e1))
            n1 = ttnn.multiply(x, r1)
            ttnn.deallocate(x)
            a = ttnn.multiply(n1, w_tt)
            r2 = ttnn.rsqrt(ttnn.add(mean(ttnn.multiply(a, a)), e2))
            bb = ttnn.multiply(a, r2)
            ttnn.deallocate(a)
            Wl, Ll, Gl, Bl = leaves
            if Bl.requires_grad:
                Bl.add_grad(ag._sum_leading(g, Bl.value.shape))
            if Gl.requires_grad:
                Gl.add_grad(ag._sum_leading(ttnn.multiply(g, bb), Gl.value.shape))
            db = ttnn.multiply(g, g_tt)
            da = ttnn.multiply(ttnn.subtract(db, ttnn.multiply(bb, mean(ttnn.multiply(db, bb)))), r2)
            for t in (db, bb, r2):
                ttnn.deallocate(t)
            if Ll.requires_grad:
                Ll.add_grad(ag._sum_leading(ttnn.multiply(da, n1), Ll.value.shape))
            if Wl.requires_grad:
                dn1 = ttnn.multiply(da, w_tt)
                dx = ttnn.multiply(ttnn.subtract(dn1, ttnn.multiply(n1, mean(ttnn.multiply(dn1, n1)))), r1)
                dW = ttnn.matmul(ttnn.reshape(dx, [n * n, c]), F, transpose_a=True, dtype=ttnn.float32,
                                 compute_kernel_config=cfg)
                Wl.add_grad(ttnn.slice(dW, [0, 0], [c, k]) if kp != k else dW)
                for t in (dn1, dx):
                    ttnn.deallocate(t)
            for t in (da, n1, r1, F, w_tt, g_tt):
                ttnn.deallocate(t)
        return bw

    return ag._tape(z, leaves, make, reads=())

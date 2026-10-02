#!/usr/bin/env python3
"""Run `tt_bio.af2_reference` in float64 at 800 tokens inside 30 GB of host RAM.

The 288-token grade (`perf/bcw_land/stack_grade.py`) builds its float64 reference by just
running the reference blocks under autograd. That stops working well before 800: a single
triangle-attention logits tensor is `n*h*n*n` float64, which is 0.71 GB at 288 and **15.26 GB
at 800**, and the graph holds two of them (logits and softmax weights) per attention, four
attentions per block. The outer-product mean's `(n, n, 1024)` is another 4.88 GB and the pair
transition's `fc1` another 2.44 GB.

`lowmem()` patches the five reference modules that build an activation bigger than the pair
representation so each one runs in row chunks under a non-reentrant checkpoint. Every chunk is
a slice along a leading index that no reduction in that module crosses, so the arithmetic per
output element is the one the unpatched module does:

  * `Attention._attend`            chunk the batch (the triangle index); softmax is over `k`
  * `TriangleMultiplication`       chunk the row-local projection in, and the row-local output
                                   half out, with the `ijc` einsum whole in between
  * `ReluTransition`               chunk rows (`fc1` is 4x the pair width)
  * `OuterProductMean`             chunk the output's `i`, with `b` and the norm whole
  * `MsaRowAttentionWithPairBias`  chunk the pair-bias projection's rows

Not bit-exact by construction: a chunked `F.linear` hands BLAS a different M, so the float64
contraction over the feature axis can land a different last ulp. `check` below measures that
drift against the unpatched reference and it is ~1e-16 relative, fourteen orders below the
~0.07 rel L2 the grade reads. Nothing here changes an op, a dtype or a reduction axis.
"""
from __future__ import annotations

import contextlib

import torch
from torch.utils.checkpoint import checkpoint


def _ck(fn, *args):
    return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)


def _spans(n, chunk):
    return [(s, min(s + chunk, n)) for s in range(0, n, chunk)]


@contextlib.contextmanager
def lowmem(chunk: int = 64, floor: int = 128):
    """Patch the reference modules for the duration of the block. `floor`: leave a leading
    dimension at or below this alone, so the 288-token path and the depth-1 MSA tracks keep
    running exactly the code they ran before."""
    from tt_bio import af2_reference as R

    att, tri, trans = R.Attention._attend, R.TriangleMultiplication.forward, \
        R.ReluTransition.forward
    opm, msarow = R.OuterProductMean.forward, R.MsaRowAttentionWithPairBias.forward
    ptrack = R.PairBlock._pair_track

    def _attend(self, q_data, m_data, bias, nonbatched_bias=None):
        b = q_data.shape[0]
        if b <= floor:
            return att(self, q_data, m_data, bias, nonbatched_bias)
        part = lambda q, m, bi, nb: att(self, q, m, bi, nb)
        return torch.cat([_ck(part, q_data[s:e], m_data[s:e], bias[s:e], nonbatched_bias)
                          for s, e in _spans(b, chunk)], 0)

    def _trimul(self, z, mask):
        n = z.shape[0]
        if n <= floor:
            return tri(self, z, mask)

        def proj_in(zc, mc):
            x = self.norm_in(zc)
            p = mc.unsqueeze(-1).to(x.dtype) * self.p_in(x)
            return p * torch.sigmoid(self.g_in(x))

        proj = torch.cat([_ck(proj_in, z[s:e], mask[s:e]) for s, e in _spans(n, chunk)], 0)
        a, b = proj.split(self.hidden, dim=-1)
        act = torch.einsum("kic,kjc->ijc" if self.ending else "ikc,jkc->ijc", a, b)
        del proj, a, b

        def out(zc, ac):
            return self.p_out(self.norm_out(ac)) * torch.sigmoid(self.g_out(self.norm_in(zc)))

        return torch.cat([_ck(out, z[s:e], act[s:e]) for s, e in _spans(n, chunk)], 0)

    def _trans(self, x):
        n = x.shape[0]
        if n <= floor:
            return trans(self, x)
        part = lambda c: trans(self, c)
        return torch.cat([_ck(part, x[s:e]) for s, e in _spans(n, chunk)], 0)

    def _opm(self, msa, mask):
        n = msa.shape[1]
        if n <= floor:
            return opm(self, msa, mask)
        mask_ = mask.unsqueeze(-1).to(msa.dtype)
        x = self.norm(msa)
        a = mask_ * self.proj_a(x)
        b = mask_ * self.proj_b(x)
        norm = torch.einsum("sic,sjc->ijc", mask_, mask_)

        def rows(ac, nc):
            outer = torch.einsum("sic,sje->ijce", ac, b).reshape(ac.shape[1], n, -1)
            return self.proj_o(outer) / (self.eps + nc)

        return torch.cat([_ck(rows, a[:, s:e], norm[s:e]) for s, e in _spans(n, chunk)], 0)

    def _pair_track(self, pair, pair_mask):
        """Checkpoint each of the five pair modules, so a block's recompute peak is one
        module's internals and not all five at once. The residual stream still costs one pair
        tensor per module; everything inside each module is recomputed."""
        if pair.shape[0] <= floor:
            return ptrack(self, pair, pair_mask)
        order = ((self.tri_mul_out, self.tri_mul_in, self.tri_att_start, self.tri_att_end)
                 if self.evoformer_order
                 else (self.tri_att_start, self.tri_att_end, self.tri_mul_out, self.tri_mul_in))
        for module in order:
            pair = pair + _ck(lambda p, m, mod=module: mod(p, m), pair, pair_mask)
        return pair + _ck(lambda p: self.pair_transition(p), pair)

    def _msarow(self, msa, msa_mask, pair):
        n = pair.shape[0]
        if n <= floor:
            return msarow(self, msa, msa_mask, pair)
        bias = (1e9 * (msa_mask - 1.0)).to(msa.dtype)[:, None, None, :]
        x = self.layer_norm(msa)
        part = lambda p: self.linear(self.pair_norm(p))
        nb = torch.cat([_ck(part, pair[s:e]) for s, e in _spans(n, chunk)], 0).permute(2, 0, 1)
        return self._attend(x, x, bias, nb)

    R.Attention._attend = _attend
    R.Attention.forward = _attend          # `Attention.forward = _attend` at class scope
    R.TriangleMultiplication.forward = _trimul
    R.ReluTransition.forward = _trans
    R.OuterProductMean.forward = _opm
    R.MsaRowAttentionWithPairBias.forward = _msarow
    R.PairBlock._pair_track = _pair_track
    R.PairBlock.forward = _pair_track
    try:
        yield
    finally:
        R.Attention._attend = att
        R.Attention.forward = att
        R.TriangleMultiplication.forward = tri
        R.ReluTransition.forward = trans
        R.OuterProductMean.forward = opm
        R.MsaRowAttentionWithPairBias.forward = msarow
        R.PairBlock._pair_track = ptrack
        R.PairBlock.forward = ptrack


def check(n=128, chunk=16, params=None, seed=0):
    """float64 forward and VJP of one Evoformer block, patched against unpatched, same inputs.

    Returns the worst relative deviation over (msa out, pair out, dm, dz). Run with `floor`
    below `n` so the patched path is actually exercised at a size the unpatched one can hold.
    """
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from perf.bcx_afgrad import afgrad as A

    torch.manual_seed(seed)
    _, ref = A.load_models(params or A.DEFAULT_PARAMS, device_arm=False)
    mod = ref["f64"]
    logits = torch.randn(n, 20) * 2.0
    with torch.no_grad():
        msa, pair = A.embed(mod, logits.double(), torch.arange(n))
    gm = torch.randn(msa.shape, dtype=torch.float64) / msa.numel() ** 0.5
    gz = torch.randn(pair.shape, dtype=torch.float64) / pair.numel() ** 0.5

    def run():
        g, outs = A.ref_vjp(lambda a, b: A.ref_evo(mod, 0, a, b), [msa, pair], [gm, gz])
        return [t.detach() for t in outs] + g

    plain = run()
    with lowmem(chunk=chunk, floor=0):
        patched = run()
    names = ["fwd_m", "fwd_z", "dm", "dz"]
    out = {}
    for name, p, q in zip(names, plain, patched):
        out[name] = {"rel": A.rel_l2(q, p), "exact": bool(torch.equal(p, q))}
    return out


if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=16)
    a = ap.parse_args()
    print(json.dumps(check(n=a.n, chunk=a.chunk), indent=1))

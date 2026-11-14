"""The flow network's initial pair representation, built on the card.

Upstream's `PairReprBuilder` concatenates ten pair features into [b, n, n, 107], projects them to
384 without a bias, applies an RMSNorm and then an AdaLN on the time embeddings. In host fp32 that
is a handful of passes over a [b, n, n, 384] tensor every sampler step, and the result then has to
be uploaded: at 512 tokens 58-73 s of host work and a 402 MB transfer per step.

Only one of the ten features moves between steps: `xt_bb_ca_pair_dists`, the noisy CA coordinates'
pairwise distances in 30 one-hot bins. The other nine come from the input. A linear layer of a
concatenation is the sum of the linears of its parts, and a one-hot times a weight is a row of the
weight, so

    z_init = AdaLN(RMSNorm(C + W_x[bin(x_t)]), t)

where `C` is the constant features' projection, computed once a design on the host in fp32 and
kept on the card, and `W_x[bin]` picks one of 30 weight rows per pair. Each step the host computes
the bin index with upstream's own `norm` + `bucketize` (so the bins are upstream's bins) and the
AdaLN's gamma and beta, which are per design ([b, 384]); the card does the rest.

Both norms are folded into one scale per pair, and the card does no reduction. With 30 bins a
pair's `x = C + W_x[k]` takes one of 30 values, and both norms' statistics are quadratic in it:

    sum(x^2)     = |C|^2     + 2 C.W_k         + |W_k|^2
    sum(x^2 w^2) = |C w|^2   + 2 C.(W_k w^2)   + |W_k w|^2

so the host tabulates `r1 * r2` for every pair and bin once a design, in float64, and each step
picks one entry per pair. The card then only computes `(C + W_x[bin]) * scale * (w gamma) + beta`,
elementwise in fp32. On the card the two sums cost +14 ms a step at 512 and were not accurate
enough: `ttnn.sum` unpacks fp32 to tf32, rel 2.7e-4 on a sum of squares, and that moved designs
at 512 by 2.1 A against the 0.6 A bar.

The card gathers `W_x[bin]` as one matmul, onehot(bin) x [hi; mid; lo], where hi, mid and lo are
the fp32 weight's three bf16 parts (`split_bf16`) stacked along K. It is not exact: the card's
accumulation across the K tiles keeps ~1e-4 at |w| ~ 0.1 (z rel rms 1.7e-5 at 512), well under the
bf16 copy the trunk reads. Three bf16 embeddings widened and added are exact but cost 14.2 ms a step
at 512 against 5.4 ms (`perf/rfd4/glue/build_bench.py`); `ttnn.embedding` only moves bf16.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

#: The pair features a sampler step changes. Everything else in `feats_pair_repr` is read off the
#: featurized input, which is the same on every step of a design.
STEP_FEATURE = "XtBBCAPairwiseDistancesPairFeat"


def split_bf16(w: torch.Tensor) -> torch.Tensor:
    """`[3, *w.shape]` bf16 parts of an fp32 tensor that sum back to it exactly in fp32.

    Each part is the next 8 significant bits of what the previous parts left, truncated rather than
    rounded, so the parts share the value's sign and occupy disjoint bits. Any partial sum then fits
    in fp32's 24, and the total is the original bit for bit whatever order the card adds them in."""
    def top8(x):  # bf16 keeps fp32's top 16 bits: sign, exponent and 7 stored mantissa bits
        return (x.view(torch.int32) & -65536).view(torch.float32)

    w = w.float().contiguous()
    hi = top8(w)
    mid = top8(w - hi)
    lo = w - hi - mid
    return torch.stack([hi, mid, lo]).bfloat16()


@dataclass
class PairConstant:
    """What a design fixes: the constant features' projection and, per pair and bin, the scale
    both RMSNorms apply to `proj + W_x[bin]`."""

    proj: torch.Tensor  # [1 or b, n, n, pair] fp32
    scales: torch.Tensor  # [1 or b, n, n, nbins] fp32, computed in float64


@dataclass
class PairInputs:
    """What the card needs for one step's pair: the design's constant (the same object on every
    step of a design, which is how the card knows it already holds it), the step's bin index and
    its norm scale, and the AdaLN's per-design scale and shift."""

    constant: PairConstant
    bins: torch.Tensor  # [b, n, n] int64, 0..nbins-1
    scale: torch.Tensor  # [b, n, n] fp32
    gamma: torch.Tensor  # [b, pair] fp32
    beta: torch.Tensor  # [b, pair] fp32


class PairSplit:
    """`PairReprBuilder` split into the part a design fixes and the part a step moves.

    Reads the builder's own feature creators and weights, so the features stay upstream's code;
    only the order of the arithmetic changes (one fp32 sum split in two)."""

    def __init__(self, builder):
        factory = builder.init_repr_factory
        if builder.cond_factory is None or factory.ret_zero:
            raise ValueError("the pair builder has no AdaLN or no features; build it on the host")
        weight = factory.linear_out.weight.detach().float()  # [pair, 107]
        self.const_creators, cols, step = [], [], []
        start = 0
        for creator in factory.feat_creators:
            stop = start + creator.get_dim()
            if type(creator).__name__ == STEP_FEATURE:
                step.append((creator, start, stop))
            else:
                self.const_creators.append(creator)
                cols.extend(range(start, stop))
            start = stop
        if len(step) != 1:
            raise ValueError(f"expected one {STEP_FEATURE} feature, found {len(step)}")
        self.step_creator, a, b = step[0]
        self.w_const = weight[:, cols]  # [pair, 77]
        self.w_step = weight[:, a:b].t().contiguous()  # [nbins, pair]
        self.ln_weight = factory.ln_out.weight.detach().float()
        self.eps = factory.ln_out.eps
        self.builder = builder
        w = self.w_step.double()
        w2 = self.ln_weight.double() ** 2
        self._cross = torch.cat([w, w * w2]).t().contiguous()  # [pair, 2 nbins]
        self._sq = torch.stack([torch.ones_like(w2), w2], 1)  # [pair, 2]
        self._wsq = torch.stack([(w * w).sum(-1), (w * w * w2).sum(-1)])  # [2, nbins]

    @property
    def nbins(self) -> int:
        return int(self.w_step.shape[0])

    def constant(self, batch: dict) -> PairConstant:
        """The projection of every feature but the step's, and its norm scales."""
        n = batch["padding_mask"].shape[-1]
        parts = []
        for creator in self.const_creators:
            f = creator(batch)
            if f.ndim == 2:  # per-example features broadcast over the pair, as FeatureFactory does
                f = f.reshape(f.shape[0], 1, 1, -1).expand(-1, n, n, -1)
            parts.append(f.float())
        proj = torch.cat(parts, dim=-1) @ self.w_const.t()
        return PairConstant(proj, self.scales(proj))

    def scales(self, proj: torch.Tensor, rows: int = 4096) -> torch.Tensor:
        """`[..., nbins]` fp32: `r1 * r2` of `proj + W_x[k]` for every bin k, in float64.
        `r1 = rsqrt(mean(x^2) + eps1)` is the first RMSNorm's, `r2 = rsqrt(r1^2 mean(x^2 w^2) +
        eps2)` the AdaLN's, which sees the first norm's output `x r1 w`."""
        c, nb = proj.shape[-1], self.nbins
        flat = proj.reshape(-1, c)
        out = torch.empty(flat.shape[0], nb)
        for i in range(0, flat.shape[0], rows):  # slices that stay in cache; 800 MB whole at 512
            x = flat[i:i + rows].double()
            cross = x @ self._cross
            sq = (x * x) @ self._sq
            r1sq = 1.0 / ((sq[:, :1] + 2 * cross[:, :nb] + self._wsq[0]) / c + self.eps)
            s2 = sq[:, 1:] + 2 * cross[:, nb:] + self._wsq[1]
            r2sq = 1.0 / (r1sq * s2 / c + self.builder.adaln.norm.eps)
            out[i:i + rows] = (r1sq * r2sq).sqrt().float()
        return out.reshape(*proj.shape[:-1], nb)

    def bins(self, batch: dict) -> torch.Tensor:
        """`[b, n, n]` int64: `bin_pairwise_distances`' bucket index, without its one-hot."""
        c = self.step_creator
        x = batch["x_t"]["bb_ca"]
        d = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)
        return torch.bucketize(d, torch.linspace(c.min_dist, c.max_dist, c.dim - 1, device=x.device))

    def modulation(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """The AdaLN's `(gamma, beta)`, `[b, pair]` each: one per design, not per pair."""
        ada = self.builder.adaln
        nc = ada.norm_cond(self.builder.cond_factory(batch))
        return ada.to_gamma(nc).float(), ada.to_beta(nc).float()

    def inputs(self, batch: dict, constant: PairConstant, modulation=None) -> PairInputs:
        gamma, beta = self.modulation(batch) if modulation is None else modulation
        bins = self.bins(batch)
        scale = torch.take_along_dim(constant.scales, bins[..., None], -1).squeeze(-1)
        return PairInputs(constant, bins, scale, gamma, beta)

    def assemble(self, p: PairInputs) -> torch.Tensor:
        """The host form of what the card computes, in float64. Unmasked: the trunk masks its
        pair on entry."""
        x = p.constant.proj.double() + self.w_step.double()[p.bins]
        wg = self.ln_weight.double() * p.gamma.double()
        return x * p.scale.double()[..., None] * wg[:, None, None] + p.beta.double()[:, None, None]


class PairOnCard:
    """Builds the trunk's fp32 `z_init` on the card from :class:`PairInputs`, padded to `npad`.

    Holds the split weights for as long as the engine lives and one design's constant
    projection for as long as that design runs."""

    def __init__(self, split: PairSplit, device):
        import ttnn

        self.split, self.device = split, device
        # bin i of part j is row 32 j + i of the stacked parts, so a bin's one-hot row picks all three
        nb, k = split.nbins, 32
        assert nb <= k, nb
        onehot, stacked = torch.zeros(nb, 3 * k), torch.zeros(3 * k, split.w_step.shape[-1])
        for j, part in enumerate(split_bf16(split.w_step).float()):
            onehot[torch.arange(nb), j * k + torch.arange(nb)] = 1.0
            stacked[j * k:j * k + nb] = part
        self.onehot = ttnn.from_torch(onehot.bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                      dtype=ttnn.bfloat16)
        self.stacked = ttnn.from_torch(stacked.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device,
                                       dtype=ttnn.bfloat16)
        # W_x itself, float32 row-major, for `tt_bio.pair_build`'s exact gather
        self.table = ttnn.from_torch(split.w_step.float().contiguous(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                     dtype=ttnn.float32)
        self._const = None  # (host PairConstant, npad, its padded projection on the card)

    def _constant(self, c: PairConstant, npad: int):
        """The design's constant projection on the card, uploaded on its first step only."""
        import ttnn

        from ..tenstorrent import upload

        if self._const is not None and self._const[0] is c and self._const[1] == npad:
            return self._const[2]
        self.release()
        proj = c.proj
        padded = torch.nn.functional.pad(proj, [0, 0, 0, npad, 0, npad]) if npad else proj
        self._const = (c, npad, upload(padded.float().contiguous(), dtype=ttnn.float32, device=self.device))
        return self._const[2]

    def __call__(self, p: PairInputs, npad: int):
        import ttnn

        from ..tenstorrent import upload

        const = self._constant(p.constant, npad)
        bins, scale = p.bins, p.scale
        if npad:
            bins = torch.nn.functional.pad(bins, [0, npad, 0, npad])
            scale = torch.nn.functional.pad(scale, [0, npad, 0, npad])
        z = self.one_program(const, bins, scale, p.gamma, p.beta)
        if z is not None:
            return z
        rows = self.rows(bins)
        x = ttnn.add(rows, const)
        ttnn.deallocate(rows)
        # z = x * r1 r2 * (w gamma) + beta; r1 r2 is the host's per-pair scale.
        s = upload(scale[..., None].float().contiguous(), dtype=ttnn.float32, device=self.device)
        wg = (self.split.ln_weight.double() * p.gamma.double()).float()
        wg, beta = (ttnn.from_torch(v[:, None, None], layout=ttnn.TILE_LAYOUT, device=self.device,
                                    dtype=ttnn.float32) for v in (wg, p.beta))
        y = ttnn.multiply(x, s)
        ttnn.deallocate(x)
        ttnn.deallocate(s)
        x = ttnn.multiply(y, wg)
        ttnn.deallocate(y)
        z = ttnn.add(x, beta)
        for t in (x, wg, beta):
            ttnn.deallocate(t)
        return z

    def one_program(self, const, bins: torch.Tensor, scale: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
        """The same z as the ttnn ops below, as one program (`tt_bio.pair_build`), or None where it declines.
        Its gather is exact, so z sits closer to float64 than the one-hot matmul's."""
        import ttnn

        from ..pair_build import pair_build

        b, n, c = int(bins.shape[0]), int(bins.shape[1]), int(self.table.shape[1])
        rm = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        bins_tt = ttnn.from_torch(bins.reshape(b * n, n).to(torch.int32), dtype=ttnn.uint32, **rm)
        scale_tt = ttnn.from_torch(scale.reshape(b * n, n).float().contiguous(), dtype=ttnn.float32, **rm)
        wg = (self.split.ln_weight.double() * gamma.double()).float()
        wg_tt, beta_tt = (ttnn.from_torch(v.float()[:, None, :].expand(b, 32, c).contiguous(), layout=ttnn.TILE_LAYOUT,
                                          device=self.device, dtype=ttnn.float32) for v in (wg, beta))
        z = pair_build(const, self.table, bins_tt, scale_tt, wg_tt, beta_tt)
        for t in (bins_tt, scale_tt, wg_tt, beta_tt):
            ttnn.deallocate(t)
        return z

    def rows(self, bins: torch.Tensor):
        """`W_x[bins]` on the card, [b, n, n, pair] fp32, the fp32 weight rows to ~1e-4."""
        import ttnn

        from .trunk import compute_kernel_config

        b, n = int(bins.shape[0]), int(bins.shape[1])
        idx = ttnn.from_torch(bins.reshape(1, b * n * n).to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=self.device, dtype=ttnn.uint32)
        onehot = ttnn.embedding(idx, self.onehot, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(idx)
        rows = ttnn.matmul(onehot, self.stacked, dtype=ttnn.float32, compute_kernel_config=compute_kernel_config())
        ttnn.deallocate(onehot)
        return ttnn.reshape(rows, (b, n, n, rows.shape[-1]))

    def release(self):
        if self._const is not None and self._const[2].is_allocated():
            import ttnn

            ttnn.deallocate(self._const[2])
        self._const = None

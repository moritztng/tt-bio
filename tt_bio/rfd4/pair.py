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

The card's gather is exact. `ttnn.embedding` only moves bf16, so each fp32 weight row is split
into three bf16 parts whose sum is the fp32 value; the card gathers each part (a row copy), widens
it to fp32 and adds the three (`tests/test_rfd4_pair_device.py`). A one-hot matmul against the
stacked parts looks equivalent but is not: with the parts stacked along K the card's accumulation
keeps only about bf16's precision, ~1e-4 off at |w| ~ 0.1.
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
class PairInputs:
    """What the card needs for one step's pair: the design's constant projection (the same object
    on every step of a design, which is how the card knows it already holds it), the step's bin
    index and the AdaLN's per-design scale and shift."""

    constant: torch.Tensor  # [1 or b, n, n, pair] fp32
    bins: torch.Tensor  # [b, n, n] int64, 0..nbins-1
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

    @property
    def nbins(self) -> int:
        return int(self.w_step.shape[0])

    def constant(self, batch: dict) -> torch.Tensor:
        """`[b, n, n, pair]` fp32: the projection of every feature but the step's."""
        n = batch["padding_mask"].shape[-1]
        parts = []
        for creator in self.const_creators:
            f = creator(batch)
            if f.ndim == 2:  # per-example features broadcast over the pair, as FeatureFactory does
                f = f.reshape(f.shape[0], 1, 1, -1).expand(-1, n, n, -1)
            parts.append(f.float())
        feats = torch.cat(parts, dim=-1)
        return feats @ self.w_const.t()

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

    def inputs(self, batch: dict, constant: torch.Tensor) -> PairInputs:
        gamma, beta = self.modulation(batch)
        return PairInputs(constant, self.bins(batch), gamma, beta)

    def assemble(self, p: PairInputs) -> torch.Tensor:
        """The host form of what the card computes, in the inputs' dtype (float64 for a reference).
        Unmasked: the trunk masks its pair on entry."""
        dt = p.constant.dtype
        x = p.constant + self.w_step.to(dt)[p.bins]
        x = torch.nn.functional.rms_norm(x, (x.shape[-1],), self.ln_weight.to(dt), self.eps)
        x = torch.nn.functional.rms_norm(x, (x.shape[-1],), None, self.builder.adaln.norm.eps)
        return x * p.gamma.to(dt)[:, None, None] + p.beta.to(dt)[:, None, None]


class PairOnCard:
    """Builds the trunk's fp32 `z_init` on the card from :class:`PairInputs`, padded to `npad`.

    Holds the split weights for as long as the engine lives and one design's constant
    projection for as long as that design runs."""

    def __init__(self, split: PairSplit, device, ckc):
        import ttnn

        self.split, self.device, self.ckc = split, device, ckc
        self.parts = [ttnn.from_torch(w, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
                      for w in split_bf16(split.w_step)]
        self.ln_w = ttnn.from_torch(split.ln_weight, layout=ttnn.TILE_LAYOUT, device=device,
                                    dtype=ttnn.float32)
        self._const = None  # (host tensor, npad, its padded copy on the card)

    def _constant(self, c: torch.Tensor, npad: int):
        """The design's constant projection on the card, uploaded on its first step only."""
        import ttnn

        from ..tenstorrent import upload

        if self._const is not None and self._const[0] is c and self._const[1] == npad:
            return self._const[2]
        self.release()
        padded = torch.nn.functional.pad(c, [0, 0, 0, npad, 0, npad]) if npad else c
        self._const = (c, npad, upload(padded.float().contiguous(), dtype=ttnn.float32, device=self.device))
        return self._const[2]

    def __call__(self, p: PairInputs, npad: int):
        import ttnn

        ckc = self.ckc
        const = self._constant(p.constant, npad)
        bins = torch.nn.functional.pad(p.bins, [0, npad, 0, npad]) if npad else p.bins
        rows = self.rows(bins)
        x = ttnn.add(rows, const)
        ttnn.deallocate(rows)
        y = ttnn.rms_norm(x, weight=self.ln_w, epsilon=self.split.eps, compute_kernel_config=ckc)
        ttnn.deallocate(x)
        x = ttnn.rms_norm(y, epsilon=self.split.builder.adaln.norm.eps, compute_kernel_config=ckc)
        ttnn.deallocate(y)
        gamma, beta = (ttnn.from_torch(v[:, None, None], layout=ttnn.TILE_LAYOUT, device=self.device,
                                       dtype=ttnn.float32) for v in (p.gamma, p.beta))
        y = ttnn.multiply(x, gamma)
        z = ttnn.add(y, beta)
        for t in (x, y, gamma, beta):
            ttnn.deallocate(t)
        return z

    def rows(self, bins: torch.Tensor):
        """`W_x[bins]` on the card, [b, n, n, pair] fp32, exactly the fp32 weight rows."""
        import ttnn

        b, n = int(bins.shape[0]), int(bins.shape[1])
        idx = ttnn.from_torch(bins.reshape(1, b * n * n).to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=self.device, dtype=ttnn.uint32)
        rows = None
        for w in self.parts:
            r = ttnn.embedding(idx, w, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            r32 = ttnn.typecast(r, ttnn.float32)
            ttnn.deallocate(r)
            if rows is None:
                rows = r32
            else:
                acc = ttnn.add(rows, r32)
                ttnn.deallocate(rows)
                ttnn.deallocate(r32)
                rows = acc
        ttnn.deallocate(idx)
        return ttnn.reshape(rows, (b, n, n, rows.shape[-1]))

    def release(self):
        if self._const is not None and self._const[2].is_allocated():
            import ttnn

            ttnn.deallocate(self._const[2])
        self._const = None

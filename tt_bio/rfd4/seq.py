"""`init_repr_factory` split into the columns a design fixes and the two a step moves.

The sequence feature factory of the released RFD4-Proteina net is 21 features wide, and exactly two
of them read the noisy state: `xt_bb_ca` and `xt_local_latents`. The other 19 (token type, the
conditioning coordinates, secondary structure, hbond, hotspot, chain centroid, ...) are functions of
the featurized example alone, yet upstream rebuilds all 21 and projects the whole concatenation on
every one of a design's 60 sampler steps.

`FeatureFactory.forward` is `mask(W cat(features))` with `W` bias-free and `ln_out` the identity
here (`use_ln_out=False`), so the projection splits along its input columns:

    W cat(f_const, f_step) = W_const f_const + W_step f_step

and the first term is the same tensor on every step of a design. This is the same split
`tt_bio.rfd4.pair.PairSplit` already makes for the pair builder, in the same fp32: only the order of
one sum changes, and the padding mask is 0/1 so masking the parts is masking the concatenation.
"""
from __future__ import annotations

import torch

#: The feature creators that read `batch["x_t"]`. Every other creator in `feats_seq` is a function
#: of the featurized example, which a design fixes.
STEP_FEATURES = ("XtBBCASeqFeat", "XtLocalLatentsSeqFeat")


class SeqSplit:
    """The fixed half of `init_repr_factory`, computed once a design instead of once a step."""

    def __init__(self, factory):
        if getattr(factory, "ret_zero", False):
            raise ValueError("the sequence factory has no features")
        if not isinstance(factory.ln_out, torch.nn.Identity):
            raise ValueError("the sequence factory normalises its projection; the split is not a sum")
        weight = factory.linear_out.weight.detach()  # [token_dim, dim_f]
        self.const_creators, self.step_creators = [], []
        const_cols, step_cols = [], []
        start = 0
        for creator in factory.feat_creators:
            stop = start + creator.get_dim()
            if type(creator).__name__ in STEP_FEATURES:
                self.step_creators.append(creator)
                step_cols.extend(range(start, stop))
            else:
                self.const_creators.append(creator)
                const_cols.extend(range(start, stop))
            start = stop
        if not self.step_creators:
            raise ValueError("the sequence factory has no state feature; it is constant whole")
        self.factory = factory
        self.w_const = weight[:, const_cols].contiguous()
        self.w_step = weight[:, step_cols].contiguous()

    def _project(self, creators, weight, batch: dict) -> torch.Tensor:
        n = batch["padding_mask"].shape[-1]
        parts = []
        for creator in creators:
            f = creator(batch)
            if f.ndim == 2:  # per-example features broadcast over the tokens, as FeatureFactory does
                f = f.reshape(f.shape[0], 1, -1).expand(-1, n, -1)
            parts.append(f)
        feats = self.factory.apply_padding_mask(torch.cat(parts, dim=-1), batch["padding_mask"])
        return torch.nn.functional.linear(feats, weight)

    def constant(self, batch: dict) -> torch.Tensor:
        """`[b, n, token_dim]`: the 19 fixed features, projected. Valid for the whole design."""
        with torch.no_grad():
            return self._project(self.const_creators, self.w_const, batch)

    def s_init(self, batch: dict, constant: torch.Tensor) -> torch.Tensor:
        """This step's `init_repr_factory(batch)`, given the design's `constant`."""
        with torch.no_grad():
            projected = self._project(self.step_creators, self.w_step, batch) + constant
            return self.factory.apply_padding_mask(projected, batch["padding_mask"])

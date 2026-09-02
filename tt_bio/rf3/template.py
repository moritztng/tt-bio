"""RF3 template embedder on ttnn.

Token-level templating: a noisy ground-truth distogram plus its noise level, run
through a small pairformer and folded back into the pair representation.

Split follows the playbook's conditioning pattern. The template features depend
only on ``f``, never on ``Z``, so they are recycle-invariant: built once on host
(an O(I^2 * 66) concat and a scalar log transform, cheap next to the pairformer)
and uploaded once. Only ``emb_pair(norm(Z)) + template_channels`` onwards runs per
recycle.

Two things scouted from the reference rather than assumed
(`rf3/model/layers/pairformer_layers.py::RF3TemplateEmbedder`):

- Its pairformer is TWO DISTINCT blocks, unlike the MSA module's single
  weight-shared one. `recycler.template_embedder.pairformer.{0,1}`.
- The blocks are c=64 with 4 heads, so triangle-attention dims are (64, 4), not the
  trunk's (32, 4). They use the `tri_mul_*` spelling, so the Pairformer renames apply.

`configs/model/components/rf3_net.yaml` says `raw_template_dim: 108`; the shipped
checkpoint's `emb_templ` is [64, 66] (64 distogram bins + has-condition + noise
level). The checkpoint is the source of truth.
"""

from __future__ import annotations


import torch
import ttnn

from tt_bio.envflags import env_flag
from tt_bio.tenstorrent import (Module, PairformerLayer, Weights,
                                sdpa_ragged_pad_site)

C = 64            # template channel width
C_Z = 128
TRI_ATT_HEAD_DIM, TRI_ATT_N_HEADS = 64, 4
N_BLOCK = 2

#: AF3 noise scale (Angstroms) -> noise level, t = (log(t_hat / 16) + 1.2) / 1.5.
_SIGMA_DATA = 16.0
_SHIFT, _SCALE = 1.2, 1.5


def noise_scale_to_level(noise_scale: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """t = (log(clamp(t_hat, min=eps) / 16) + 1.2) / 1.5.

    The clamp is BEFORE the division, not an epsilon added after it. For a zero
    noise scale the two differ by 1.8 in t (-13.3 against -11.5), which is worth
    0.006 of PCC on a templated input.
    """
    return (torch.log(torch.clamp(noise_scale, min=eps) / _SIGMA_DATA)
            + _SHIFT) / _SCALE


def template_features(f: dict) -> torch.Tensor:
    """Build the [I, I, 66] template feature block on host. Recycle-invariant."""
    has_dc = f["has_distogram_condition"].float()          # [I, I]
    scale = f["distogram_condition_noise_scale"].float()   # [I]
    distogram = f["distogram_condition"].float()           # [I, I, 64]

    joint_scale = (scale[None, :] ** 2 + scale[:, None] ** 2).sqrt()
    joint_level = noise_scale_to_level(joint_scale)

    feats = torch.cat(
        [distogram, has_dc.unsqueeze(-1), joint_level.unsqueeze(-1)], dim=-1
    )
    # zero out interactions with no condition -- masking AFTER the concat, so the
    # has-condition and noise-level channels are masked too
    return feats * has_dc.unsqueeze(-1)


#: Route the template embedder's triangle attention through the fused SDPA, as the trunk
#: stack and the confidence head already do. This is the site that sets RF3's Wormhole
#: ceiling: it runs one block per token over the raw token axis, so the materialised
#: fp32-softmax chain writes [tokens, heads, S_pad, S_pad]. At 656 tokens that is
#: 656 x 4 x 672 x 672 x 2 = 2369912832 B, which is the exact request the 656 aa fold died
#: on one second into `trunk 0/10` (measured 2026-09-02 on GWH02, and again on
#: wk/ceiling-rf3 after the MSA module alone had been moved). 650 aa asks 2348236800 B,
#: the same tensor six tokens narrower.
#:
#: `sdpa_ragged_pad` ships with it and never without: this block sees the raw token axis,
#: so the key tail is ragged whenever the input is not a multiple of 32.
#:
#: Set TT_BIO_RF3_TEMPLATE_FUSED_SDPA=0 for the old materialised route.
_TEMPLATE_FUSED_SDPA = env_flag("TT_BIO_RF3_TEMPLATE_FUSED_SDPA", True)


class TemplateEmbedder(Module):
    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.emb_templ_weight = self.torch_to_tt("emb_templ.weight")
        self.emb_pair_weight = self.torch_to_tt("emb_pair.weight")
        self.agg_emb_weight = self.torch_to_tt("agg_emb.weight")
        self.pre_norm_weight = self.torch_to_tt("norm_pair_before_pairformer.weight")
        self.pre_norm_bias = self.torch_to_tt("norm_pair_before_pairformer.bias")
        self.post_norm_weight = self.torch_to_tt("norm_after_pairformer.weight")
        self.post_norm_bias = self.torch_to_tt("norm_after_pairformer.bias")
        self.blocks = [
            PairformerLayer(
                TRI_ATT_HEAD_DIM, TRI_ATT_N_HEADS, None, None, False,
                self.scope(f"pairformer.{i}"), compute_kernel_config,
                scale_pair_bias=False, transpose_bias=False,
                # fp32_softmax=True was "the reference runs its triangle-attention softmax
                # in fp32", and it was worth 60-85% relative RMS -- against the fused route
                # with an UNMASKED ragged key tail, which is the comparison
                # `remap.PAIRFORMER_FLAGS` re-ran and reversed for the trunk. Masked, the
                # fused route is the accurate one and it does not write the score tensor.
                # See `_TEMPLATE_FUSED_SDPA`.
                fp32_softmax=not _TEMPLATE_FUSED_SDPA,
                tri_att_sdpa_ragged_pad=(
                    _TEMPLATE_FUSED_SDPA and sdpa_ragged_pad_site("rf3.tri_att", True)),
            )
            for i in range(N_BLOCK)
        ]

    def embed_template_feats(self, feats: ttnn.Tensor) -> ttnn.Tensor:
        """Project the host-built [I, I, 66] block to [I, I, C]. Call once."""
        return ttnn.linear(
            feats, self.emb_templ_weight,
            compute_kernel_config=self.compute_kernel_config,
        )

    def __call__(
        self,
        z: ttnn.Tensor,
        template_channels: ttnn.Tensor,
        mask: ttnn.Tensor | None = None,
        attn_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        z_norm = ttnn.layer_norm(
            z, weight=self.pre_norm_weight, bias=self.pre_norm_bias, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        v = ttnn.linear(
            z_norm, self.emb_pair_weight,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(z_norm)
        v = ttnn.add_(v, template_channels)

        for block in self.blocks:
            v = block(None, v, mask=mask,
                      attn_mask_start=attn_mask, attn_mask_end=attn_mask)[1]

        # upstream adds this to a zeros tensor, so the norm output is the result
        v = ttnn.layer_norm(
            v, weight=self.post_norm_weight, bias=self.post_norm_bias, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        v = ttnn.relu(v)
        out = ttnn.linear(
            v, self.agg_emb_weight,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(v)
        return out

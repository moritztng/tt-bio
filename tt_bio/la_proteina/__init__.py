# La-Proteina — non-equivariant all-atom protein generation (port in progress)

# SPDX-License-Identifier: Apache-2.0
#
# La-Proteina reference code (NVIDIA-Digital-Bio/la-proteina) is Apache-2.0,
# vendored under _vendor/la-proteina-ref.
# La-Proteina weights are under the NVIDIA Open Model License (NOML, Apr 28 2025):
#   - Models are commercially usable.
#   - Derivative Models may be created and distributed.
#   - NVIDIA claims no ownership of outputs.
# See tt_bio/la_proteina/NOTICE for the NOML attribution required on distribution.

"""La-Proteina port — pass 3.

Pass 1 cleared the license gate + scoped the architecture. Pass 2 vendored the
reference, built the component-level golden harness, and ported the denoiser's
core attention block (TTPairBiasAttentionAdaLN). Pass 3 (this branch,
wk/tt-bio-la-proteina-port-p3) extends pass-2's component-by-component discipline
to the rest of the denoiser trunk, same random-weight PCC bar (>= 0.999), golden
= unmodified vendored reference:

  - TTTransition            : plain SwiGLU transition (the conditioning path
                              transition_c_1/2 and the inner block of TransitionADALN).
  - TTTransitionADALN       : AdaptiveLayerNorm + Transition + AdaptiveOutputScale.
  - TTLocalLatentsHead / TTCaHead : the two output heads (LN + Linear -> 8 / -> 3).
  - TTMultiheadAttnAndTransition : one trunk layer (attn + transition, sequential,
                              both residual) — the stitch of the pass-2 attention
                              block + TransitionADALN.
  - TTPairReprUpdate (no tri-mult) : pair-representation update (outer-product-style
                              pair bias injection + openfold PairTransition). The
                              use_tri_mult=True path reuses tt_bio.tenstorrent.
                              TriangleMultiplication (same openfold state-dict
                              layout) and is deferred to a follow-on pass.

Parity (160M denoiser dims, B=1 N=64, bf16, HiFi4 + fp32_dest_acc, random weights,
identical to the golden) — scripts/la_proteina_trunk_parity.py:

  component                                  all-True   partial
  TransitionADALN                            0.999972   0.999977
  conditioning (transition_c_1/2)            0.999960   0.999948
  head: local_latents_linear                 0.999995   0.999995
  head: ca_linear                            0.999995   0.999995
  MultiheadAttnAndTransition (layer)         0.999997   0.999996
  PairReprUpdate (no tri-mult)               ~1.000      ~1.000

The pass-2 attention block was refactored to share the AdaLN/AdaptiveOutputScale
helpers with the new blocks and re-verified: all-True 0.999723, partial 0.999764
(unchanged from pass 2).

NOT done (pass 4+): real-weight parity (still blocked on NGC checkpoint access),
the full multi-layer LocalLatentsTransformer forward + Euler sampler loop, the
autoencoder, and the tri-mult pair-update path. See docs/la-proteina-port.md and
~/.coworker/notes/tt-bio-la-proteina-port-p3.md.
"""

from tt_bio.la_proteina.denoiser import (
    TTPairBiasAttentionAdaLN,
    TTTransition,
    TTTransitionADALN,
    TTLocalLatentsHead,
    TTCaHead,
    TTMultiheadAttnAndTransition,
    TTPairReprUpdate,
    TTPairTransition,
)

__all__ = [
    "TTPairBiasAttentionAdaLN",
    "TTTransition",
    "TTTransitionADALN",
    "TTLocalLatentsHead",
    "TTCaHead",
    "TTMultiheadAttnAndTransition",
    "TTPairReprUpdate",
    "TTPairTransition",
]

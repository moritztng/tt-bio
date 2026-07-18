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

"""La-Proteina port — pass 2.

Pass 1 (branch wk/tt-bio-la-proteina-port) cleared the license gate, confirmed
the param count (~160M denoiser / ~130M AE encoder+decoder / ~420M total), and
scoped the architecture. Pass 2 (this branch, wk/tt-bio-la-proteina-port-p2):

  - Vendored the reference implementation (NVIDIA-Digital-Bio/la-proteina,
    Apache-2.0) under _vendor/la-proteina-ref.
  - Built a component-level PyTorch golden harness (scripts/la_proteina_attn_parity.py)
    that runs the unmodified vendored denoiser attention block on fixed inputs and
    compares against the ttnn port — the tt-bio port-parity idiom.
  - Ported the denoiser's core sequence-side attention block
    `MultiHeadBiasedAttentionADALN_MM` (AdaptiveLayerNorm + PairBiasAttention with
    QK-LN + pair bias + gated output + AdaptiveOutputScale) to ttnn
    (denoiser.TTPairBiasAttentionAdaLN), mirroring tt_bio/tenstorrent.py's
    AttentionPairBias/AdaLN and adding the La-Proteina-specific QK-LN and
    cond-conditioned AdaptiveOutputScale.

Parity (160M denoiser dims, B=1 N=64, bf16, HiFi4 + fp32_dest_acc, random
weights, identical to the golden):
  - all-True mask : PCC 0.999723 (>= 0.999)  PASS
  - partial mask  : PCC 0.999764 (>= 0.999)  PASS

Real-weight parity is blocked on NGC checkpoint access (the NGC catalog serves
the .ckpt via a browser-auth file-browser, not a direct download URL); that plus
the full LocalLatentsTransformer trunk (PairReprUpdate + tri-mult + TransitionADALN
+ the Euler sampler loop) and the autoencoder are pass 3+. See
docs/la-proteina-port.md and ~/.coworker/notes/tt-bio-la-proteina-port-p2.md.
"""

from tt_bio.la_proteina.denoiser import TTPairBiasAttentionAdaLN

__all__ = ["TTPairBiasAttentionAdaLN"]

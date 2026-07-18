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

"""La-Proteina port — pass 4.

Passes 1-3 cleared the license gate, vendored the reference, built the
component-level golden harness, and ported the denoiser trunk component-by-
component (attention block, TransitionADALN, conditioning, both output heads,
a single trunk layer, the non-tri-mult PairReprUpdate), all at the random-weight
PCC bar (>= 0.999) against the unmodified vendored reference.

Pass 4 (this branch, wk/tt-bio-la-proteina-port-p4) extends that to the full
trunk forward and the remaining denoiser + AE surfaces, same bar:

  - TTTransformerTrunk : shared denoiser/AE trunk orchestrator (cond stack +
                         nlayers x MultiheadAttnAndTransition + optional
                         PairReprUpdate). TTLocalLatentsTransformer (the full
                         denoiser) now delegates to it.
  - TTLocalLatentsTransformer : the FULL 14-layer denoiser trunk (160M config,
                         update_pair_repr=False) + cond + both heads, and the
                         160M_tri config (update_pair_repr=True, every_n=2,
                         use_tri_mult=True). Error does NOT compound below 0.999
                         over 14 layers.
  - TTTriangleMultiplicativeUpdate : self-contained direct ttnn port of openfold
                         Algorithms 11/12 (outgoing/incoming), wired into
                         TTPairReprUpdate (use_tri_mult=True). NOT reused from
                         tt_bio.tenstorrent.TriangleMultiplication because that
                         port uses the boltz2/protenix state-dict key layout, not
                         openfold's, and is coupled to the tenstorrent.py Module
                         framework — reuse was not free.
  - TTEulerStep (sampler.py) : the flow-matching Euler integration step (all
                         four sampling modes: vf, vf_ss, sc, vf_ss_sc_sn) +
                         score<->vector-field transforms + mask + center-of-mass.
                         The stochastic `eps` draw is an explicit shared input
                         (per memory diffusion-port-parity-shared-draws).
  - TTEncoderTransformer / TTDecoderTransformer (autoencoder.py) : the AE
                         encoder (latent head, shared-eps z) and decoder (seq-logit
                         + struct-coordinate heads, abs_coors post-process).

Parity (160M denoiser + 130M AE dims, B=1 N=64, bf16, HiFi4 + fp32_dest_acc,
random weights, both all-True and partial masks):

  full trunk (160M, 14 layers)        local_latents 0.99996 / ca 0.99995
  full trunk (160M_tri, tri-mult)     local_latents 0.99996 / ca 0.99996
  PairReprUpdate (tri-mult, component) ~1.000
  Euler step (18 cases, 2 data modes) 0.999996-0.999998
  AE encoder (mean/log_scale/z)       0.99996-0.99998
  AE decoder (logits/coors)            0.99996-0.99999

NOT done (pass 5+): real-weight parity (still blocked on NGC checkpoint access),
the full nsteps Euler sampler LOOP around the denoiser (gated on the
FeatureFactory/PairReprBuilder dataset feature-pipeline port — the same gate
that blocks the full end-to-end forward), and perf (fold into tenstorrent.py,
trace capture, the perf-grade fused tri-mult kernel). No end-to-end real-weight
parity is claimed. See docs/la-proteina-port.md and
~/.coworker/notes/tt-bio-la-proteina-port-p4.md.
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
    TTTriangleMultiplicativeUpdate,
    TTTransformerTrunk,
    TTLocalLatentsTransformer,
)
from tt_bio.la_proteina.sampler import TTEulerStep
from tt_bio.la_proteina.autoencoder import (
    TTEncoderTransformer,
    TTDecoderTransformer,
    TTLatentHead,
    TTLogitHead,
    TTStructHead,
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
    "TTTriangleMultiplicativeUpdate",
    "TTTransformerTrunk",
    "TTLocalLatentsTransformer",
    "TTEulerStep",
    "TTEncoderTransformer",
    "TTDecoderTransformer",
    "TTLatentHead",
    "TTLogitHead",
    "TTStructHead",
]

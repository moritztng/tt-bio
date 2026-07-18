# La-Proteina — non-equivariant all-atom protein generation (port in progress)

# SPDX-License-Identifier: Apache-2.0
#
# La-Proteina reference code (NVIDIA-Digital-Bio/la-proteina) is Apache-2.0.
# La-Proteina weights are under the NVIDIA Open Model License (NOML, Apr 28 2025):
#   - Models are commercially usable.
#   - Derivative Models may be created and distributed.
#   - NVIDIA claims no ownership of outputs.
# See tt_bio/la_proteina/NOTICE for the NOML attribution required on distribution.

"""La-Proteina port — pass 1 scaffold.

Pass 1 established: license clearance (NOML — commercially usable, derivatives
allowed), confirmed param count (~160M denoiser / ~130M AE encoder+decoder /
~420M total), and architecture scoping. The flow-matching denoiser transformer
is the first port target; it reuses tt-bio's existing pair-biased-attention
trunk (boltz2 DiffusionTransformer), triangular-multiplicative-update primitive,
and BoltzGen-style Euler sampler loop. The autoencoder (encoder/decoder) is a
follow-on pass.

Status: scaffold only. No ttnn implementation or PCC parity yet. See
docs/la-proteina-port.md and ~/.coworker/notes/tt-bio-la-proteina-port-p1.md.
"""

__all__: list[str] = []

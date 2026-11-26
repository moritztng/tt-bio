"""The scalar featurisers: sigma embedding and the Gaussian distance expansions.

These are the cheapest modules in the score network and the first two that can be graded
without a golden, because both are closed forms: the grade is against a float64 recompute of
the same closed form, not against another implementation. They matter out of proportion to
their cost -- every edge in the graph carries a distance expansion, and the sigma embedding
is what tells the network where in the reverse diffusion it is.

Upstream: ``utils/diffusion_utils.sinusoidal_embedding`` and ``models/layers.GaussianSmearing``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def sinusoidal_embedding(t: Tensor, dim: int, max_positions: int = 10000) -> Tensor:
    """``[N] -> [N, dim]``, upstream's DDPM embedding, sin block then cos block.

    Note the two scales: the score model's ``embedding_scale`` is 1000 and multiplies *t*
    before this function, while ``max_positions`` stays at 10000 and sets the frequency
    spacing. The confidence model uses 10000 for both. Mixing them up changes every edge and
    node feature in the network and nothing crashes.

    **The angle is computed in float64 and cast back, which upstream does not do.** With
    ``embedding_scale`` at 1000 the largest angle is ~1000 radians, and fp32 resolves an
    angle that size to about 6e-5 rad, so upstream's fp32 ``sin``/``cos`` are wrong by
    4.7e-5 absolute on a feature whose range is [-1, 1] -- 2.2e-4 for the confidence model at
    scale 10000. In float64 the same expression lands within 3.0e-8 of the exact value, a
    factor of 1500 and 7000. The cost is one ``[N, 64]`` tensor per graph on the host, which
    is nothing, and the alternative is carrying an error larger than anything the device port
    will add into every node and edge feature in the network. Bit-exactness with upstream
    here would be bit-exactness with upstream's rounding noise.
    """
    if t.dim() != 1:
        raise ValueError(f"sinusoidal_embedding wants a 1-D tensor of times, got {tuple(t.shape)}")
    half = dim // 2
    step = math.log(max_positions) / (half - 1)
    freqs = torch.exp(torch.arange(half, dtype=torch.float64, device=t.device) * -step)
    ang = t.to(torch.float64)[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1).to(t.dtype)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def timestep_embedding(t: Tensor, dim: int, scale: int) -> Tensor:
    """``get_timestep_embedding('sinusoidal', dim, scale)`` applied to *t*."""
    return sinusoidal_embedding(scale * t, dim)


class GaussianSmearing(torch.nn.Module):
    """``[E] -> [E, num]`` radial basis, ``exp(-0.5 ((d - mu_k)/delta)^2)``.

    ``delta`` is the spacing of the centres, so the coefficient is fixed by *start*, *stop*
    and *num* and is not a learned or stored quantity: the checkpoint keeps only the centres.
    DiffDock builds four of these, with (stop, num) = (5, 64) for ligand edges, (15, 64) for
    receptor edges, (80, 64) for cross edges and (30, 64) for the centre convolution.
    """

    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians, dtype=torch.float64)
        # Upstream takes the spacing off an fp32 linspace, so its coefficient carries an
        # fp32 rounding error that the exponent then multiplies by up to ~10. Taking it in
        # float64 costs nothing (it is one scalar, fixed at construction) and keeps the whole
        # expansion within fp32 epsilon of the exact one.
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset.to(torch.float32))

    def forward(self, dist: Tensor) -> Tensor:
        d = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * d.pow(2))


def gaussian_smearing_reference(dist: Tensor, module: "GaussianSmearing") -> Tensor:
    """The float64 recompute *module* is graded against.

    It evaluates the same closed form with the same centres, in float64. Grading against
    centres recomputed in float64 instead would measure something else: the centres are fp32
    buffers in the checkpoint, and ``dist - centre`` cancels, so a centre rounded by 6e-8
    relative moves the result by ``2|coeff| |d|`` times that -- about 5e-6 at the 80 A cross
    expansion. That is a property of the stored parameters, not of the implementation, and
    folding it into the bar would hide a real arithmetic regression behind it.
    """
    offset = module.offset.to(torch.float64)
    d = dist.to(torch.float64).view(-1, 1) - offset.view(1, -1)
    return torch.exp(module.coeff * d.pow(2))

"""The rollout on the card: the fixed-shape step and its pose update as ttnn ops (D-dfd-perf-6).

An op-for-op port of the torch program that already defines a step over fixed shapes,
:func:`tt_bio.diffdock.static_net.static_forward` followed by :func:`tt_bio.diffdock.loop.diffusion_step`.
Every block here is graded against that torch twin in float64 on the card
(`scripts/diffdock_port/device_loop_grade.py`) before the next one builds on it, and nothing here
is new arithmetic.

Layout: a per-edge or per-node scalar is a logical ``[n, 1]`` TILE tensor, so every elementwise
op is fp32 on the vector engine, and a coordinate triple travels as three of them. Wide edge
features (harmonics, distance expansions) are assembled from those columns; only the embedding
MLPs touch the matrix engine.
"""

from __future__ import annotations

import math

import torch

_EPS = 1e-12          # harmonics.unit's clamp, e3nn's F.normalize default


def upload(device, x: torch.Tensor):
    """A host tensor as an fp32 TILE tensor on *device*; a 1-d tensor becomes ``[n, 1]``."""
    import ttnn
    x = x.to(torch.float32)
    if x.dim() == 1:
        x = x[:, None]
    return ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def norm3(x, y, z):
    """``|(x, y, z)|`` per row."""
    import ttnn
    return ttnn.sqrt(ttnn.add(ttnn.add(ttnn.multiply(x, x), ttnn.multiply(y, y)),
                              ttnn.multiply(z, z)))


def harmonics(x, y, z, ls=(0, 1, 2)):
    """:func:`tt_bio.diffdock.harmonics.spherical_harmonics` of ``(x, y, z)``, normalised.

    Returns ``[n, sum(2l+1)]``, e3nn's component order. A zero vector stays zero past l=0, as
    ``unit`` makes it on the host.
    """
    import ttnn
    inv = ttnn.reciprocal(ttnn.clamp(norm3(x, y, z), min=_EPS))
    x, y, z = (ttnn.multiply(c, inv) for c in (x, y, z))
    s3, s5, s15 = math.sqrt(3), math.sqrt(5), math.sqrt(15)
    cols = []
    for l in ls:
        if l == 0:
            cols.append(ttnn.add(ttnn.multiply(x, 0.0), 1.0))
        elif l == 1:
            cols += [ttnn.multiply(c, s3) for c in (x, y, z)]
        elif l == 2:
            y2 = ttnn.multiply(y, y)
            x2z2 = ttnn.add(ttnn.multiply(x, x), ttnn.multiply(z, z))
            cols += [ttnn.multiply(ttnn.multiply(x, z), s15),
                     ttnn.multiply(ttnn.multiply(x, y), s15),
                     ttnn.multiply(ttnn.subtract(y2, ttnn.multiply(x2z2, 0.5)), s5),
                     ttnn.multiply(ttnn.multiply(y, z), s15),
                     ttnn.multiply(ttnn.subtract(ttnn.multiply(z, z), ttnn.multiply(x, x)),
                                   0.5 * s15)]
        else:
            raise ValueError(f"harmonics are implemented to l=2, asked for {l}")
    return ttnn.concat(cols, dim=1)


class Smearing:
    """:class:`tt_bio.diffdock.features.GaussianSmearing` on the card: ``exp(coeff (d - mu)^2)``."""

    def __init__(self, device, module):
        self.offset = upload(device, module.offset.reshape(1, -1))
        self.coeff = float(module.coeff)

    def __call__(self, dist):
        import ttnn
        d = ttnn.subtract(dist, self.offset)
        return ttnn.exp(ttnn.multiply(ttnn.multiply(d, d), self.coeff))


class MLP:
    """An ``nn.Sequential`` of Linear / ReLU / Identity on the card, fp32 weights."""

    def __init__(self, device, seq: torch.nn.Sequential):
        self.ops = []
        for m in seq:
            if isinstance(m, torch.nn.Linear):
                self.ops.append((upload(device, m.weight.detach().t().contiguous()),
                                 upload(device, m.bias.detach().reshape(1, -1))))
            elif isinstance(m, torch.nn.ReLU):
                self.ops.append("relu")
            elif not isinstance(m, (torch.nn.Identity, torch.nn.Dropout)):
                raise NotImplementedError(f"no card form for {type(m).__name__}")

    def __call__(self, x):
        import ttnn
        for op in self.ops:
            x = ttnn.relu(x) if op == "relu" else ttnn.linear(x, op[0], bias=op[1])
        return x

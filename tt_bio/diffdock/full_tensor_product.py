"""``o3.FullTensorProduct``: every pair of input irreps, no weights.

The torsion head needs exactly one of these. Upstream builds
``o3.FullTensorProduct(sh_irreps, "2e")`` to combine each torsion edge's harmonics with the
harmonics of the rotatable bond it belongs to, and feeds the 45-wide result to
``tor_bond_conv`` as *its* harmonics. It is the only tensor product in the model with no
learned weights, and the only one whose first argument reaches ``l = 2`` -- which is why the
shipped Wigner table had to be widened (97d3465ee) and why the released checkpoint carries
``final_tp_tor._compiled_main_left_right._w3j_*`` for precisely those 9 triples.

Two differences from :class:`~tt_bio.diffdock.tensor_product.FullyConnectedTensorProduct`,
both of which change the numbers without changing any shape:

* **The output irreps are sorted**, by ``(l, p)`` with odd before even (e3nn's ``Irrep`` is a tuple), and that sort decides the column layout
  the next layer reads. e3nn enumerates paths ``in1`` then ``in2`` then sorts the collected
  outputs; the paths keep their enumeration order but write into sorted slices.
* **The path normalisation is ``sqrt(dim(ir_out))``** here, not the ``element`` rule. With
  ``uvuv`` connectivity each output channel is reached by exactly one path
  (``num_elements = 1``), so e3nn's denominator is 1 and only the ``component`` numerator
  survives. Reusing the fully-connected rule would scale every block by the wrong constant.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tt_bio.diffdock.irreps import Irrep, Irreps
from tt_bio.diffdock.tensor_product import wigner_3j


class FullTensorProduct(torch.nn.Module):
    """``(x1, x2) -> out``, with ``out`` determined by the inputs."""

    def __init__(self, in1_irreps, in2_irreps):
        super().__init__()
        self.in1_irreps = Irreps.parse(in1_irreps)
        self.in2_irreps = Irreps.parse(in2_irreps)

        collected = []          # (mul, l, p) in e3nn's enumeration order
        paths = []              # (i, j, index into `collected`)
        for i, t1 in enumerate(self.in1_irreps):
            for j, t2 in enumerate(self.in2_irreps):
                for l in range(abs(t1.l - t2.l), t1.l + t2.l + 1):
                    collected.append((t1.mul * t2.mul, l, t1.p * t2.p))
                    paths.append((i, j, len(collected) - 1))

        order = sorted(range(len(collected)), key=lambda k: (collected[k][1], collected[k][2]))
        self.out_irreps = Irreps(tuple(Irrep(collected[k][0], collected[k][1], collected[k][2])
                                        for k in order))
        slot_of = {k: pos for pos, k in enumerate(order)}
        out_slices = self.out_irreps.slices()
        self.paths = tuple((i, j, slot_of[k]) for i, j, k in paths)
        self._out_slices = [out_slices[slot_of[k]] for _, _, k in paths]
        # uvuv: one path per output channel, so e3nn's `element` denominator is 1 and the
        # whole coefficient is the `component` numerator sqrt(dim(ir_out)).
        self.path_weights = tuple(float(2 * collected[k][1] + 1) ** 0.5 for _, _, k in paths)

        self._in1_slices = self.in1_irreps.slices()
        self._in2_slices = self.in2_irreps.slices()
        self._w3j64 = [
            wigner_3j(self.in1_irreps.terms[i].l, self.in2_irreps.terms[j].l,
                      collected[k][1], dtype=torch.float64)
            for (i, j, k) in paths
        ]
        self._w3j_cast: dict = {}

    def _w3j(self, n: int, like: Tensor) -> Tensor:
        key = (n, like.dtype, like.device)
        got = self._w3j_cast.get(key)
        if got is None:
            got = self._w3j64[n].to(dtype=like.dtype, device=like.device)
            self._w3j_cast[key] = got
        return got

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        if x1.shape[-1] != self.in1_irreps.dim:
            raise ValueError(f"x1 width {x1.shape[-1]}, expected {self.in1_irreps.dim}")
        if x2.shape[-1] != self.in2_irreps.dim:
            raise ValueError(f"x2 width {x2.shape[-1]}, expected {self.in2_irreps.dim}")
        lead = x1.shape[:-1]
        out = x1.new_zeros((*lead, self.out_irreps.dim))
        for n, (i, j, k) in enumerate(self.paths):
            t1, t2 = self.in1_irreps.terms[i], self.in2_irreps.terms[j]
            a = x1[..., self._in1_slices[i]].reshape(*lead, t1.mul, 2 * t1.l + 1)
            b = x2[..., self._in2_slices[j]].reshape(*lead, t2.mul, 2 * t2.l + 1)
            part = torch.einsum("...ui,...vj,ijk->...uvk", a, b, self._w3j(n, x1))
            out[..., self._out_slices[n]] = part.reshape(
                *lead, t1.mul * t2.mul * part.shape[-1]) * self.path_weights[n]
        return out

    def extra_repr(self) -> str:
        return f"{self.in1_irreps} x {self.in2_irreps} -> {self.out_irreps}"

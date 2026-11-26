"""``o3.FullyConnectedTensorProduct`` with per-edge weights, in plain torch.

This is the hot op of the DiffDock score network: 17 of these, 4164 per-edge weights each at
the released ns=48/nv=10, once per edge per diffusion step. e3nn reaches it by compiling a
TorchScript kernel per irrep signature; tt-bio reaches it by contracting six small Wigner 3j
tensors it ships as a table, which is both portable to ttnn and readable.

What has to match e3nn exactly, and does (``tests/test_diffdock_tensor_product.py`` grades it
against e3nn 0.5.0 in the reference environment):

* **Path order.** e3nn enumerates ``in1`` outermost, then ``in2``, then ``out``, keeping any
  triple whose selection rule holds. The per-edge weights are concatenated in that order, so
  getting it wrong silently feeds every path the wrong slice of the FC block's output.
* **Path normalisation.** With e3nn's defaults (``component`` irrep normalisation,
  ``element`` path normalisation) each path is scaled by
  ``sqrt(dim(ir_out) / sum_over_paths_into_that_output(mul_in1 * mul_in2))``. It is a
  constant of the irreps, but it is not 1, and the checkpoint's weights were fitted with it.
* **The 3j sign.** A 3j tensor is unique only up to sign. The table comes from e3nn
  (``scripts/diffdock_port/gen_w3j.py``) for exactly that reason.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor

from tt_bio.diffdock.irreps import Irreps

_TABLE_PATH = Path(__file__).with_name("w3j_tables.json")


@lru_cache(maxsize=None)
def _tables() -> dict[tuple[int, int, int], Tensor]:
    raw = json.loads(_TABLE_PATH.read_text())["tables"]
    out = {}
    for key, flat in raw.items():
        l1, l2, l3 = (int(x) for x in key.split(","))
        out[(l1, l2, l3)] = torch.tensor(flat, dtype=torch.float64).reshape(
            2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)
    return out


def wigner_3j(l1: int, l2: int, l3: int, *, dtype=torch.float32, device=None) -> Tensor:
    """e3nn's normalised Wigner 3j, from the shipped table."""
    try:
        w = _tables()[(l1, l2, l3)]
    except KeyError:
        raise KeyError(
            f"no Wigner 3j table for ({l1}, {l2}, {l3}). The shipped table covers what the "
            f"released DiffDock-L reaches; regenerate with scripts/diffdock_port/gen_w3j.py "
            f"in the reference environment if a new irrep signature is genuinely needed."
        ) from None
    return w.to(dtype=dtype, device=device)


class FullyConnectedTensorProduct(torch.nn.Module):
    """``(x1, x2, w) -> out`` with the weights supplied per element, not stored.

    ``shared_weights=False`` in upstream: the weights come from an FC block over the edge
    features, so one tensor product serves every edge with a different weight.
    """

    def __init__(self, in_irreps, sh_irreps, out_irreps):
        super().__init__()
        self.in_irreps = Irreps.parse(in_irreps)
        self.sh_irreps = Irreps.parse(sh_irreps)
        self.out_irreps = Irreps.parse(out_irreps)

        paths = []
        for i, ti in enumerate(self.in_irreps):
            for j, tj in enumerate(self.sh_irreps):
                for k, tk in enumerate(self.out_irreps):
                    if ti.p * tj.p != tk.p:
                        continue
                    if not (abs(ti.l - tj.l) <= tk.l <= ti.l + tj.l):
                        continue
                    paths.append((i, j, k))
        self.paths = tuple(paths)

        # e3nn's "element" path normalisation: every path into one output irrep shares the
        # denominator, so the scale of an output does not depend on how many ways it is
        # reachable. in1_var/in2_var/out_var are all 1 here, as upstream leaves them.
        fan_in: dict[int, int] = {}
        for i, j, k in paths:
            fan_in[k] = fan_in.get(k, 0) + self.in_irreps.terms[i].mul * self.sh_irreps.terms[j].mul
        self.path_weights = tuple(
            (self.out_irreps.terms[k].dim / self.out_irreps.terms[k].mul / fan_in[k]) ** 0.5
            for i, j, k in paths
        )
        self.weight_shapes = tuple(
            (self.in_irreps.terms[i].mul, self.sh_irreps.terms[j].mul, self.out_irreps.terms[k].mul)
            for i, j, k in paths
        )
        self.weight_numel = sum(a * b * c for a, b, c in self.weight_shapes)

        self._in_slices = self.in_irreps.slices()
        self._sh_slices = self.sh_irreps.slices()
        self._out_slices = self.out_irreps.slices()
        for n, (i, j, k) in enumerate(paths):
            self.register_buffer(
                f"w3j_{n}",
                wigner_3j(self.in_irreps.terms[i].l, self.sh_irreps.terms[j].l,
                          self.out_irreps.terms[k].l),
                persistent=False,
            )

    def forward(self, x1: Tensor, x2: Tensor, weight: Tensor) -> Tensor:
        if x1.shape[-1] != self.in_irreps.dim:
            raise ValueError(f"x1 width {x1.shape[-1]}, expected {self.in_irreps.dim}")
        if x2.shape[-1] != self.sh_irreps.dim:
            raise ValueError(f"x2 width {x2.shape[-1]}, expected {self.sh_irreps.dim}")
        if weight.shape[-1] != self.weight_numel:
            raise ValueError(f"weight width {weight.shape[-1]}, expected {self.weight_numel}")

        lead = x1.shape[:-1]
        out = x1.new_zeros((*lead, self.out_irreps.dim))
        start = 0
        for n, (i, j, k) in enumerate(self.paths):
            u, v, w_mul = self.weight_shapes[n]
            ti = self.in_irreps.terms[i]
            tj = self.sh_irreps.terms[j]
            tk = self.out_irreps.terms[k]
            w = weight[..., start:start + u * v * w_mul].reshape(*lead, u, v, w_mul)
            start += u * v * w_mul

            a = x1[..., self._in_slices[i]].reshape(*lead, u, 2 * ti.l + 1)
            b = x2[..., self._sh_slices[j]].reshape(*lead, v, 2 * tj.l + 1)
            w3j = getattr(self, f"w3j_{n}").to(x1.dtype)
            part = torch.einsum("...uvw,...ui,...vj,ijk->...wk", w, a, b, w3j)
            part = part.reshape(*lead, w_mul * (2 * tk.l + 1)) * self.path_weights[n]
            out[..., self._out_slices[k]] += part
        return out

    def extra_repr(self) -> str:
        return (f"{self.in_irreps} x {self.sh_irreps} -> {self.out_irreps}, "
                f"{len(self.paths)} paths, weight_numel={self.weight_numel}")

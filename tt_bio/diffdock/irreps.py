"""The slice of e3nn's irrep algebra DiffDock actually uses, in plain torch.

DiffDock depends on ``e3nn==0.5.0``; tt-bio does not and will not. e3nn's value is its
general tensor-product compiler, and the released DiffDock-L checkpoint exercises exactly
one corner of it: ``o3.FullyConnectedTensorProduct(in, 1x0e+1x1o+1x2e, out)`` with
per-edge weights, over irreps that never go past l=1 on the feature side. Pulling in a
JIT-compiling dependency to reach that corner would also pull its codegen into every
device path. So the irrep bookkeeping lives here and the tensor product itself is a
dense contraction we own.

e3nn stays the *reference*: the parity gate grades this module against
``o3.FullyConnectedTensorProduct`` in the upstream environment, never the other way round.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# e3nn lets a term drop its multiplicity: "2e" is "1x2e".  The torsion head's second
# input is written that way upstream, so parse accepts both spellings.
_TERM = re.compile(r"^\s*(?:(\d+)\s*x\s*)?(\d+)([eo])\s*$")


@dataclass(frozen=True)
class Irrep:
    """``mul x l p``: *mul* copies of the degree-*l* representation of parity *p*."""

    mul: int
    l: int
    p: int                      # +1 even, -1 odd

    @property
    def dim(self) -> int:
        return self.mul * (2 * self.l + 1)

    def __str__(self) -> str:
        return f"{self.mul}x{self.l}{'e' if self.p > 0 else 'o'}"


@dataclass(frozen=True)
class Irreps:
    """An ordered list of :class:`Irrep`, in e3nn's string order and layout.

    Order matters: it fixes the column layout of every feature tensor and therefore the
    meaning of every checkpoint weight. Never sort or simplify these.
    """

    terms: tuple[Irrep, ...]

    @classmethod
    def parse(cls, text: str | "Irreps") -> "Irreps":
        if isinstance(text, Irreps):
            return text
        terms = []
        for part in str(text).split("+"):
            m = _TERM.match(part)
            if not m:
                raise ValueError(f"not an irreps term: {part!r} (in {text!r})")
            terms.append(Irrep(int(m.group(1) or 1), int(m.group(2)),
                               1 if m.group(3) == "e" else -1))
        if not terms:
            raise ValueError(f"empty irreps: {text!r}")
        return cls(tuple(terms))

    @property
    def dim(self) -> int:
        return sum(t.dim for t in self.terms)

    def slices(self) -> list[slice]:
        out, start = [], 0
        for t in self.terms:
            out.append(slice(start, start + t.dim))
            start += t.dim
        return out

    def index(self, l: int, p: int) -> int | None:
        for i, t in enumerate(self.terms):
            if t.l == l and t.p == p:
                return i
        return None

    def mul_of(self, l: int, p: int) -> int:
        i = self.index(l, p)
        return 0 if i is None else self.terms[i].mul

    def __iter__(self):
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def __str__(self) -> str:
        return " + ".join(str(t) for t in self.terms)


def spherical_harmonics_irreps(lmax: int) -> Irreps:
    """``o3.Irreps.spherical_harmonics(lmax)``: parity ``(-1)**l``, one copy each."""
    return Irreps(tuple(Irrep(1, l, (-1) ** l) for l in range(lmax + 1)))


def irrep_seq(ns: int, nv: int, use_second_order_repr: bool, reduce_pseudoscalars: bool) -> list[str]:
    """``models.tensor_layers.get_irrep_seq``, verbatim.

    The conv stack walks this list: layer *i* maps ``seq[min(i, -1)] -> seq[min(i+1, -1)]``,
    so after the third layer every layer is irreps-stationary.
    """
    if use_second_order_repr:
        return [
            f"{ns}x0e",
            f"{ns}x0e + {nv}x1o + {nv}x2e",
            f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o",
            f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + "
            f"{nv if reduce_pseudoscalars else ns}x0o",
        ]
    return [
        f"{ns}x0e",
        f"{ns}x0e + {nv}x1o",
        f"{ns}x0e + {nv}x1o + {nv}x1e",
        f"{ns}x0e + {nv}x1o + {nv}x1e + {nv if reduce_pseudoscalars else ns}x0o",
    ]


def irrep_to_size(irrep: str | Irreps) -> int:
    return Irreps.parse(irrep).dim


def fully_connected_paths(
    in_irreps: str | Irreps, sh_irreps: str | Irreps, out_irreps: str | Irreps
) -> list[tuple[int, int, int, int]]:
    """Every ``(i_in, i_sh, i_out, path_weight_numel)`` of a FullyConnectedTensorProduct.

    e3nn opens a ``uvw`` path for each ``(in_i, sh_j, out_k)`` whose selection rule holds:
    ``|l_i - l_j| <= l_k <= l_i + l_j`` and ``p_i * p_j == p_k``. Each path carries
    ``mul_i * mul_j * mul_k`` weights, laid out in that order, and the weights of all paths
    are concatenated in iteration order (in, then sh, then out). ``weight_numel`` is their
    sum, and the FC block that produces the per-edge weights emits exactly that many
    numbers, so this ordering is load-bearing for weight loading.
    """
    a, b, c = Irreps.parse(in_irreps), Irreps.parse(sh_irreps), Irreps.parse(out_irreps)
    paths = []
    for i, ti in enumerate(a):
        for j, tj in enumerate(b):
            for k, tk in enumerate(c):
                if ti.p * tj.p != tk.p:
                    continue
                if not (abs(ti.l - tj.l) <= tk.l <= ti.l + tj.l):
                    continue
                paths.append((i, j, k, ti.mul * tj.mul * tk.mul))
    return paths


def weight_numel(in_irreps, sh_irreps, out_irreps) -> int:
    return sum(n for *_, n in fully_connected_paths(in_irreps, sh_irreps, out_irreps))

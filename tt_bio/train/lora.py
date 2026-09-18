"""LoRA at Tier 2: the config, the factors, the composed forward, and site discovery.

``lora_factors`` works on shapes and is the piece that shipped. ``lora_factors_for`` is the
model-aware wrapper, and it is the one that needs A1's attach point: there is no registry of
adaptable sites to look up, so it finds them by **running the shipped forward once with a
census hook installed**. Every call that routes through ``tt_bio.ops.linear`` announces itself
with its own shapes; a call that does not route through it is not adaptable, and the census
says so by not listing it. That is a discovery mechanism rather than a hand-maintained list,
which matters because a hand-maintained list is wrong the first time a module is retuned.

Sites are named by call site -- ``file:line:qualname`` of the first frame above ``ops`` -- and
not by weight identity, because the same weight object is read at one site while two different
weights are read at one shared helper. The call site is the thing a user can find and target.

``attach`` composes over whatever hook is already installed instead of replacing it. A frozen
trunk still has to stay taped downstream of its first adapter or the gradient never reaches
the adapters in the early layers, so declining a non-adapter site outright would silently
train only the last block. The delegation is through ``ops.grad_hook()``, which is public.
"""

from __future__ import annotations

import math
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import ttnn

from .. import autograd as ag
from .. import ops
from .tensors import to_device

__all__ = ["LoraConfig", "LoraSite", "lora_factors", "lora_factors_for", "lora_linear",
           "census", "select", "attach"]


@dataclass
class LoraConfig:
    """``ttml.modules.LoraConfig``'s fields, minus the ones we have no mechanism for.

    ``lora_dropout`` is deliberately absent: the tape has no dropout op, and adding one
    would need a reproducible on-device RNG whose gradient is a mask. tt-train's SFT
    example runs 0.05 on 500 steps of Shakespeare; a handful of structures is a different
    regime and rank is the regulariser here.

    ``targets`` is a tuple of regular expressions matched against a site NAME as
    :func:`census` reports it (``file:line:qualname``). Empty means every site the census
    found, which is the honest default for a discovery-based adapter: a silent subset is how
    a fine-tune reproduces the base model's loss and gets read as "LoRA does nothing".
    """

    rank: int = 8
    alpha: float = 16.0
    use_rslora: bool = False
    targets: tuple = ()

    @property
    def scaling(self) -> float:
        """``alpha/rank``, or ``alpha/sqrt(rank)`` under rank-stabilised LoRA."""
        return (self.alpha / math.sqrt(self.rank)) if self.use_rslora else (self.alpha / self.rank)


def lora_factors(in_features: int, out_features: int, cfg: LoraConfig, device, *,
                 dtype=ttnn.bfloat16, rng=None):
    """``(A, B)`` in ttnn's ``(in, out)`` layout, both taped and requiring gradients.

    tt-train stores torch-style ``A (rank, in)`` and ``B (out, rank)``; ttnn's linear takes
    ``(in, out)``, so these are ``A (in, rank)`` and ``B (rank, out)``. A is uniform on
    ``+/- 1/sqrt(in_features)`` (``_create_lora_A``, lora.py:56-60) and B is exactly zero
    (``_create_lora_B``, :64-67).
    """
    import numpy as np
    rng = np.random.default_rng() if rng is None else rng
    bound = 1.0 / math.sqrt(in_features)
    a = rng.uniform(-bound, bound, size=(in_features, cfg.rank)).astype(np.float32)
    b = np.zeros((cfg.rank, out_features), dtype=np.float32)
    return (ag.Tensor(to_device(a, device, dtype=dtype), requires_grad=True),
            ag.Tensor(to_device(b, device, dtype=dtype), requires_grad=True))


def lora_linear(x: ag.Tensor, w: ag.Tensor, a: ag.Tensor, b: ag.Tensor,
                bias: Optional[ag.Tensor] = None, *, scaling: float, config=None) -> ag.Tensor:
    """``linear(x, w, bias) + linear(linear(x, a), b) * scaling``.

    Composed entirely from the tape's own ``linear``, ``add`` and ``scale``, so dA and dB
    are produced by ``linear``'s already-gradchecked weight rule rather than by a new
    backward. ``w`` keeps ``requires_grad = False``, which prunes its branch in ``_tape``
    and so costs nothing to carry -- tt-train freezes it the same way
    (``lora.py:92``).
    """
    base = ag.linear(x, w, bias, config=config)
    down = ag.linear(x, a, config=config)
    up = ag.linear(down, b, config=config)
    return ag.add(base, ag.scale(up, scaling))



# ------------------------------------------------------------------ site discovery

@dataclass(frozen=True)
class LoraSite:
    """One adaptable ``ops.linear`` call, as the census found it.

    ``in_features``/``out_features`` come off the weight's own shape rather than off ``x``,
    because ``x`` arrives in whatever rank the caller had and the weight is always ``(in,
    out)`` by tt-bio's convention. ``calls`` is how many times the site fired in one forward,
    which is the number that says whether a site is per-block or shared -- a shared site takes
    one adapter that every block reads, and adapting it is a different model from adapting
    each block's own.
    """

    name: str
    in_features: int
    out_features: int
    calls: int = 1
    has_bias: bool = False
    activation: Optional[str] = None

    def __str__(self) -> str:
        act = f" +{self.activation}" if self.activation else ""
        return (f"{self.name}  ({self.in_features} -> {self.out_features}"
                f"{', bias' if self.has_bias else ''}{act})  x{self.calls}")


# The frames to skip when naming a site: the dispatch itself and this module. Named by file
# suffix rather than by module object so a reload cannot desynchronise it.
_INTERNAL = ("tt_bio/ops.py", "tt_bio/train/lora.py", "tt_bio/autograd.py")


def _site_name() -> str:
    """``file:line:qualname`` of the first frame above the dispatch machinery.

    ``sys._getframe`` rather than ``traceback``: this runs once per linear call during a
    census, and building a formatted traceback per call turned a 230-token forward into
    something nobody would wait for.
    """
    f = sys._getframe(1)
    while f is not None:
        name = f.f_code.co_filename.replace("\\", "/")
        if not any(name.endswith(s) for s in _INTERNAL):
            short = "/".join(name.split("/")[-2:])
            return f"{short}:{f.f_lineno}:{f.f_code.co_name}"
        f = f.f_back
    return "<unknown>"


class _Census:
    """A grad hook that records every ``ops.linear`` it sees and declines all of them.

    Declining is what makes it safe to run on the real forward: ``ops.linear`` falls through
    to ``shipped_linear``, so a census pass computes exactly what an inference pass computes
    and the output is comparable byte for byte. It is a listener, not a substitute.
    """

    def __init__(self):
        self.sites: Dict[str, LoraSite] = {}

    def linear(self, x, w, bias, *, activation=None, **kw):
        name = _site_name()
        shape = tuple(w.shape)
        prev = self.sites.get(name)
        if prev is None:
            self.sites[name] = LoraSite(name=name, in_features=int(shape[-2]),
                                        out_features=int(shape[-1]),
                                        has_bias=bias is not None, activation=activation)
        elif (prev.in_features, prev.out_features) != (int(shape[-2]), int(shape[-1])):
            # One call site reached with two different weight shapes. An adapter keyed to the
            # site would be the wrong shape for one of them, and picking either silently is
            # how a run trains a factor that never matches its operand.
            raise ValueError(
                f"site {name} was reached with two weight shapes: "
                f"{prev.in_features}x{prev.out_features} and {shape[-2]}x{shape[-1]}. "
                f"Split the call site or target the callers instead")
        else:
            self.sites[name] = LoraSite(
                name=name, in_features=prev.in_features, out_features=prev.out_features,
                calls=prev.calls + 1, has_bias=prev.has_bias, activation=prev.activation)
        return None

    def layer_norm(self, x, weight, bias, **kw):
        # Layer norm has no low-rank factorisation to add: its parameters are one vector per
        # feature, so a rank-r decomposition of them is larger than they are.
        return None


def census(forward, *args, **kwargs) -> Dict[str, LoraSite]:
    """Run ``forward`` once with a recording hook and return the adaptable sites by name.

    Restores whatever hook was installed, including none, so a census inside a training run
    does not disarm the tape. The forward's own output is discarded: what is wanted is the
    list, and a census that also returned a structure would invite using it as a prediction
    made under a hook nobody audited.
    """
    rec = _Census()
    prev = ops.set_grad_hook(rec)
    try:
        with ag.no_grad():
            forward(*args, **kwargs)
    finally:
        ops.set_grad_hook(prev)
    return dict(rec.sites)


def select(sites: Dict[str, LoraSite], cfg: LoraConfig) -> Dict[str, LoraSite]:
    """The sites ``cfg.targets`` selects. Empty ``targets`` selects all of them.

    A pattern that matches nothing raises rather than being dropped. A typo in a target is
    otherwise indistinguishable from a site that no longer exists, and both come out as a
    fine-tune that quietly adapted less than it was told to.
    """
    if not cfg.targets:
        return dict(sites)
    out, missed = {}, []
    for pat in cfg.targets:
        hit = {n: s for n, s in sites.items() if re.search(pat, n)}
        if not hit:
            missed.append(pat)
        out.update(hit)
    if missed:
        raise ValueError(f"targets matched no site: {missed}. Sites found: "
                         f"{sorted(sites)}")
    return out


def lora_factors_for(forward, cfg: LoraConfig, device, *args, dtype=ttnn.bfloat16,
                     rng=None, **kwargs) -> Dict[str, Tuple[ag.Tensor, ag.Tensor]]:
    """Discover the adaptable sites in ``forward`` and build ``(A, B)`` for each selected one.

    This is ``lora_factors`` lifted from shapes to a model. It runs the forward once, which
    costs one inference, and that is the price of not maintaining a list of site names by
    hand. Returns ``{site_name: (A, B)}``, ready to hand to :func:`attach` and to name the
    optimizer's parameters with.
    """
    sites = select(census(forward, *args, **kwargs), cfg)
    if not sites:
        raise ValueError(
            "the census found no adaptable linear site. Either the forward does not route "
            "through tt_bio.ops.linear -- in which case it is not adaptable without routing "
            "it, which is A1's attach work and not a config change -- or it was not actually "
            "called")
    return {name: lora_factors(s.in_features, s.out_features, cfg, device,
                               dtype=dtype, rng=rng)
            for name, s in sites.items()}


@contextmanager
def attach(factors: Dict[str, Tuple[ag.Tensor, ag.Tensor]], cfg: LoraConfig):
    """Install the adapters for the duration of the block, composing over the current hook.

    The delegation is the load-bearing part. At a site with no adapter this hands the call to
    whatever hook was already installed -- ``tt_bio.autograd``'s tape, normally -- so the
    trunk downstream of the first adapter stays taped and the gradient reaches the adapters in
    the early layers. Declining instead would run those sites untaped and train only whatever
    sits after the last adapter, with no error and a loss curve that still falls.
    """
    scaling = cfg.scaling

    class _Adapter:
        def __init__(self, base):
            self.base = base

        def linear(self, x, w, bias, *, activation=None, compute_kernel_config=None, **kw):
            pair = factors.get(_site_name())
            if pair is None:
                return None if self.base is None else self.base.linear(
                    x, w, bias, activation=activation,
                    compute_kernel_config=compute_kernel_config, **kw)
            if activation is not None:
                # `ttnn.linear` fuses the activation into the packer, so `base + up*scaling`
                # would be added AFTER it rather than inside it -- a different function, not a
                # different precision. Refused rather than approximated.
                raise NotImplementedError(
                    f"site {_site_name()} fuses activation {activation!r}; a LoRA sum has to "
                    f"land inside the activation, not after it. Target a site without one")
            a, b = pair
            return lora_linear(ag.Tensor(x) if not isinstance(x, ag.Tensor) else x,
                               ag.Tensor(w) if not isinstance(w, ag.Tensor) else w,
                               a, b,
                               None if bias is None else
                               (bias if isinstance(bias, ag.Tensor) else ag.Tensor(bias)),
                               scaling=scaling, config=compute_kernel_config)

        def layer_norm(self, x, weight, bias, **kw):
            return None if self.base is None else self.base.layer_norm(x, weight, bias, **kw)

    prev = ops.grad_hook()
    ops.set_grad_hook(_Adapter(prev))
    try:
        yield factors
    finally:
        ops.set_grad_hook(prev)

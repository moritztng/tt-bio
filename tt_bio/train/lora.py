"""LoRA at Tier 2: the config, the factors, the composed forward, and site discovery.

``lora_factors`` works on shapes and is the piece that shipped. ``lora_factors_for`` is the
model-aware wrapper, and it is the one that needs A1's attach point: there is no registry of
adaptable sites to look up, so it finds them by **running the shipped forward once with a
census hook installed**. Every call that routes through ``tt_bio.ops.linear`` announces itself
with its own shapes; a call that does not route through it is not adaptable, and the census
says so by not listing it. That is a discovery mechanism rather than a hand-maintained list,
which matters because a hand-maintained list is wrong the first time a module is retuned.

Sites are named by call site -- ``file:line:qualname`` of the first frame above the dispatch -- and
not by weight identity, because the same weight object is read at one site while two different
weights are read at one shared helper. The call site is the thing a user can find and target.

``attach`` composes over whatever hook is already installed instead of replacing it. A frozen
trunk still has to stay taped downstream of its first adapter or the gradient never reaches
the adapters in the early layers, so declining a non-adapter site outright would silently
train only the last block. The delegation is through ``ops.grad_hook()``, which is public.
"""

from __future__ import annotations

import hashlib
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
           "census", "select", "attach", "weights_for", "trainable", "Parameters"]


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
    # An int is a seed, a Generator is the caller's own stream, None is fresh entropy. The
    # int form exists because an unseeded init is unreproducible, and unreproducible is not
    # survivable under data parallelism: see `_site_rng`.
    rng = (rng if isinstance(rng, np.random.Generator)
           else np.random.default_rng() if rng is None else np.random.default_rng(rng))
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
#
# `dispatch.py` is on this list because the hook is no longer called from `ops.py`. The op
# offers itself to the hook from inside `OpSurface.dispatching`'s wrapper, so that wrapper's
# frame sits between the real caller and us; without it every site names itself
# `tt_bio/dispatch.py:<line>:call` and the census collapses to one entry.
_INTERNAL = ("tt_bio/ops.py", "tt_bio/dispatch.py", "tt_bio/train/lora.py",
             "tt_bio/autograd.py")


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


def _linear_operands(args, kwargs):
    """``(x, w, bias)`` from a dispatched ``ops.linear`` call, however it was passed.

    The hook is handed the call's ``args`` and ``kwargs`` untouched, and ``bias`` is
    positional at some sites and a keyword at others, so it is bound by position first and
    name second. Same binding `tt_bio.autograd._taped_linear` does, for the same reason.
    """
    args = list(args) + [None] * (3 - len(args))
    return args[0], args[1], (args[2] if args[2] is not None else kwargs.get("bias"))


class _Census:
    """A grad hook that records every ``ops.linear`` it sees and declines all of them.

    Declining is what makes it safe to run on the real forward: the dispatch falls through to
    ``linear.shipped``, so a census pass computes exactly what an inference pass computes and
    the output is comparable byte for byte. It is a listener, not a substitute.

    Only ``linear`` is recorded. Layer norm has no low-rank factorisation to add -- its
    parameters are one vector per feature, so a rank-r decomposition of them is larger than
    they are -- and it declines along with every other op by falling through the name test.
    """

    def __init__(self):
        self.sites: Dict[str, LoraSite] = {}

    def __call__(self, name, shipped, args, kwargs):
        if name != "linear":
            return None
        _, w, bias = _linear_operands(args, kwargs)
        activation = kwargs.get("activation")
        site = _site_name()
        shape = tuple(w.shape)
        prev = self.sites.get(site)
        if prev is None:
            self.sites[site] = LoraSite(name=site, in_features=int(shape[-2]),
                                        out_features=int(shape[-1]),
                                        has_bias=bias is not None, activation=activation)
        elif (prev.in_features, prev.out_features) != (int(shape[-2]), int(shape[-1])):
            # One call site reached with two different weight shapes. An adapter keyed to the
            # site would be the wrong shape for one of them, and picking either silently is
            # how a run trains a factor that never matches its operand.
            raise ValueError(
                f"site {site} was reached with two weight shapes: "
                f"{prev.in_features}x{prev.out_features} and {shape[-2]}x{shape[-1]}. "
                f"Split the call site or target the callers instead")
        else:
            self.sites[site] = LoraSite(
                name=site, in_features=prev.in_features, out_features=prev.out_features,
                calls=prev.calls + 1, has_bias=prev.has_bias, activation=prev.activation)
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


def _site_rng(rng, name: str):
    """One stream per SITE, so the adapter init does not depend on discovery order.

    A seed spread over the site names rather than consumed in whatever order the census
    returned them. That distinction is cheap here and expensive later: a data-parallel run
    builds the same adapters once per chip, and two ranks whose censuses came back in a
    different order would start from different weights, train two different models, and show
    two loss curves that both fall.

    A ``Generator`` is used as handed in, because then the caller owns the stream. ``None``
    stays fresh entropy, which is the honest answer for a caller that asked for no seed --
    and it is why every entry point that has a seed passes it.
    """
    import numpy as np
    if isinstance(rng, np.random.Generator) or rng is None:
        return rng
    tag = int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "big")
    return np.random.default_rng([int(rng), tag])


def lora_factors_for(forward, cfg: LoraConfig, device, *args, dtype=ttnn.bfloat16,
                     rng=None, **kwargs) -> Dict[str, Tuple[ag.Tensor, ag.Tensor]]:
    """Discover the adaptable sites in ``forward`` and build ``(A, B)`` for each selected one.

    This is ``lora_factors`` lifted from shapes to a model. It runs the forward once, which
    costs one inference, and that is the price of not maintaining a list of site names by
    hand. Returns ``{site_name: (A, B)}``, ready to hand to :func:`attach` and to name the
    optimizer's parameters with.

    ``rng`` may be an int seed, a ``Generator``, or ``None`` for fresh entropy. Pass the seed:
    A is random and B is zero, so an unseeded init means two processes adapting the same model
    start from different weights. Under the launcher that is one process per chip, which makes
    it a silent divergence rather than an unreproducible run.
    """
    sites = select(census(forward, *args, **kwargs), cfg)
    if not sites:
        raise ValueError(
            "the census found no adaptable linear site. Either the forward does not route "
            "through tt_bio.ops.linear -- in which case it is not adaptable without routing "
            "it, which is A1's attach work and not a config change -- or it was not actually "
            "called")
    return {name: lora_factors(s.in_features, s.out_features, cfg, device,
                               dtype=dtype, rng=_site_rng(rng, name))
            for name, s in sites.items()}


@contextmanager
def attach(installed, cfg: Optional[LoraConfig] = None):
    """Install what this run trains, for the duration of the block, over the current hook.

    Takes what :func:`trainable` returned, and which of the two it is comes from ``cfg``:
    ``None`` means ``installed`` is the model's own weights and each site gets a taped view of
    its own; a :class:`LoraConfig` means ``installed`` is ``{site: (A, B)}`` and each site gets
    a factor pair added beside a weight that does not move.

    The delegation is the load-bearing part in both modes. At a site this run does not train,
    the call goes to whatever hook was already installed -- ``tt_bio.autograd``'s tape,
    normally -- so the trunk downstream of the first trained site stays taped and the gradient
    reaches the sites in the early layers. Declining instead would run those sites untaped and
    train only whatever sits after the last one, with no error and a loss curve that still
    falls.
    """
    if cfg is None:
        prev = ops.grad_hook()
        ops.set_grad_hook(_Substitute(installed, prev, frozen=True))
        try:
            yield installed
        finally:
            ops.set_grad_hook(prev)
        return

    factors = installed
    scaling = cfg.scaling

    class _Adapter:
        """Adapt the selected ``linear`` sites; hand everything else to ``base`` verbatim.

        Every op that is not an adapted linear -- layer norm, an unselected site, a future
        op this module has never heard of -- is delegated with the arguments exactly as the
        dispatch passed them, so composing over the tape cannot drop a keyword the way
        re-spelling the signature here once could.
        """

        def __init__(self, base):
            self.base = base

        def __call__(self, name, shipped, args, kwargs):
            pair = factors.get(_site_name()) if name == "linear" else None
            if pair is None:
                return None if self.base is None else self.base(name, shipped, args, kwargs)
            x, w, bias = _linear_operands(args, kwargs)
            activation = kwargs.get("activation")
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
                               scaling=scaling,
                               config=kwargs.get("compute_kernel_config"))

    prev = ops.grad_hook()
    ops.set_grad_hook(_Adapter(prev))
    try:
        yield factors
    finally:
        ops.set_grad_hook(prev)


# ------------------------------------------------- the model's own weights, same sites
#
# Full-weight training and LoRA differ in exactly ONE thing: what the optimizer owns. So they
# share the census, the loop, the optimizer, the checkpointer and the data-parallel axis, and
# the difference reaches a user as one argument (`lora=None`) rather than as a second recipe.
# `trainable()` below is the single call the recipe makes, and it is why the Tier-1 body is
# byte-identical for a LoRA fine-tune and a full pre-training run.


class Parameters(dict):
    """``{name: Tensor}``, plus which weight each name was discovered on.

    A plain dict everywhere it is consumed -- the optimizer, the checkpointer and the
    provenance all take it as one -- with the discovery's ``(site, id(weight)) -> name`` map
    carried alongside. That map is what lets :func:`attach` tell a weight it has seen from one
    it has not, and a forward that re-uploads its weights is exactly a forward whose weights
    are never the ones discovery saw. Rebuilding the map from the names would not work: the
    names are a deterministic function of call order, so a re-uploading forward mints the
    SAME names for different tensors and looks identical.

    A hand-built plain dict is still accepted by ``attach``; it simply has no map to check
    against, so the re-upload check is skipped rather than failing on an honest Tier-2 user.
    """

    __slots__ = ("origin",)

    def __init__(self, *args, origin=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin = {} if origin is None else origin


class _Substitute:
    """A grad hook that replaces each selected site's weight with a taped parameter.

    LoRA adds a factor pair beside the weight and leaves the weight alone. This hands the
    forward a taped view OF the weight, so ``autograd._taped_linear`` accumulates into it and
    ``AdamW.step`` writes the updated value back into ``t.value``, which is what the next
    forward then reads.

    The weight the model passes is IGNORED after the first sighting of it. It has to be: the
    optimizer replaces ``t.value`` with a new device tensor every step, while the model's own
    cache still holds the tensor it uploaded once. Reading the passed weight would train a
    parameter nobody reads and leave the forward on the checkpoint's weights for the whole
    run, with a falling loss curve and no error anywhere.

    Names are minted per ``(site, distinct weight)`` and not per site, which is the difference
    from the LoRA hook and is not a detail. ``_site_name`` is ``file:line:qualname``, so a
    48-block trunk whose blocks run the same line collapses to ONE site name carrying 48
    DIFFERENT weights. One adapter every block reads is a modelling choice; training block 0's
    weight and applying it to all 48 is not a choice, it is wrong.
    """

    def __init__(self, params: Dict[str, ag.Tensor], base=None, *,
                 targets: Tuple[str, ...] = (), frozen: bool = False):
        self.params = params
        self.base = base
        self.targets = tuple(targets)
        # True once the parameter set is the optimizer's. A weight discovery never saw is then
        # a forward re-uploading its weights, which would train tensors it never reads again --
        # gradients real, optimizer stepping, model not moving.
        self.frozen = frozen
        self._key: Dict[Tuple[str, int], str] = dict(getattr(params, "origin", None) or {})
        self._checked = bool(self._key)

    def _name(self, site: str, w) -> str:
        """This weight's parameter name. ``site`` bare, then ``site#1``, ``site#2``, ...

        Numbered by first appearance within a forward, which is deterministic for a given
        program: every data-parallel rank walks the same blocks in the same order off the same
        checkpoint, so rank 3's ``site#17`` is rank 0's ``site#17``.
        """
        k = (site, id(w))
        name = self._key.get(k)
        if name is None:
            if self._checked:
                # The map came from discovery and this tensor is not in it. The forward did
                # not keep the weights it was discovered on.
                raise ValueError(
                    f"site {site} was reached with a weight discovery never saw, so this "
                    f"forward does not hold its device weights between calls. Full-weight "
                    f"training needs weights uploaded once and reused: against a re-uploading "
                    f"forward the optimizer owns tensors the forward never reads again, and "
                    f"the run looks healthy while the model stands still. Cache the weights, "
                    f"or train adapters, which are held by the run rather than by the model")
            n = sum(1 for (s, _) in self._key if s == site)
            name = site if n == 0 else f"{site}#{n}"
            self._key[k] = name
        return name

    def _param(self, name: str, w) -> ag.Tensor:
        t = self.params.get(name)
        if t is None:
            if self.frozen:
                raise ValueError(f"{name!r} is not one of the parameters this run trains; "
                                 f"the forward changed between discovery and the loop")
            t = self.params[name] = ag.Tensor(w, requires_grad=True)
        elif tuple(t.value.shape) != tuple(w.shape):
            raise ValueError(f"{name!r} was discovered at {tuple(t.value.shape)} and reached "
                             f"again at {tuple(w.shape)}; one name cannot be two weights")
        return t

    def _delegate(self, name, shipped, args, kwargs):
        """Hand the call to whatever hook was installed under us.

        Delegation rather than a call to ``shipped``, for the reason ``attach`` delegates: the
        hook below is the tape, and it is what knows the backward. Declining instead would run
        the substituted site untaped -- a parameter the optimizer owns that no gradient ever
        reaches.
        """
        return None if self.base is None else self.base(name, shipped, args, kwargs)

    def __call__(self, name, shipped, args, kwargs):
        if name != "linear":
            return self._delegate(name, shipped, args, kwargs)
        site = _site_name()
        if self.targets and not any(re.search(p, site) for p in self.targets):
            return self._delegate(name, shipped, args, kwargs)
        _, w, _ = _linear_operands(args, kwargs)
        if isinstance(w, ag.Tensor):        # already a parameter; nothing to substitute
            return self._delegate(name, shipped, args, kwargs)
        t = self._param(self._name(site, w), w)
        # The weight slot, however the caller spelled it. Every shipped site passes it
        # positionally, but binding by position alone would silently skip a keyword caller and
        # train a parameter the forward never reads.
        if len(args) >= 2:
            args = tuple(t if i == 1 else a for i, a in enumerate(args))
        elif "w" in kwargs:
            kwargs = {**kwargs, "w": t}
        else:
            raise TypeError(f"site {site} called ops.linear with no weight argument to "
                            f"substitute; args={len(args)}, kwargs={sorted(kwargs)}")
        return self._delegate(name, shipped, args, kwargs)


def weights_for(forward, cfg: Optional[LoraConfig], *args, **kwargs) -> Dict[str, ag.Tensor]:
    """The model's OWN weights at the adaptable sites, as taped parameters.

    One forward under a collecting hook -- the same one forward ``lora_factors_for`` spends on
    its census, and for the same reason: the parameter set comes from the model rather than
    from a list somebody maintains by hand. ``cfg.targets`` selects and an empty ``targets``
    takes every site; ``cfg=None`` is every site with no LoRA config to read.

    No seed argument, and that asymmetry with ``lora_factors_for`` is the point. A LoRA factor
    is INITIALISED here, so an unseeded init means two ranks train different models. A weight
    is not initialised here at all -- it comes off the checkpoint every rank loaded -- so
    there is no randomness to pin and a seed parameter would imply one.
    """
    params = Parameters()
    prev = ops.grad_hook()
    hook = _Substitute(params, prev, targets=tuple(cfg.targets) if cfg else ())
    ops.set_grad_hook(hook)
    try:
        with ag.no_grad():
            forward(*args, **kwargs)
    finally:
        ops.set_grad_hook(prev)
    params.origin = hook._key
    if not params:
        raise ValueError(
            "the census found no adaptable linear site. Either the forward does not route "
            "through tt_bio.ops.linear -- in which case it is not trainable without routing "
            "it, which is attach work and not a config change -- or it was not actually "
            "called")
    return params


def trainable(forward, cfg: Optional[LoraConfig], device, *args, rng=None, **kwargs):
    """What this run trains, from one discovery forward. ``(installed, params)``.

    The whole of the LoRA / full-weight choice, in one call, so the Tier-1 body does not
    branch on it and the two runs are demonstrably the same program:

    * ``cfg`` a :class:`LoraConfig` -- ``installed`` is ``{site: (A, B)}`` and ``params`` is
      those factors under ``site.A`` / ``site.B`` names. The trunk is frozen.
    * ``cfg`` ``None`` -- ``installed`` and ``params`` are both the model's own weights at
      those same sites. Nothing is frozen, and this is what a pre-training run passes.

    Hand ``installed`` to :func:`attach` and ``params`` to the optimizer, in that order.
    """
    if cfg is None:
        params = weights_for(forward, None, *args, **kwargs)
        return params, params
    factors = lora_factors_for(forward, cfg, device, *args, rng=rng, **kwargs)
    return factors, {f"{site}.{which}": t
                     for site, pair in factors.items()
                     for which, t in zip(("A", "B"), pair)}

"""The starting weights for the ABodyBuilder3 reproduction, as upstream draws them.

`tt_bio.af2_reference.Linear` allocates zeros and records the name of the initializer its call
site asked for; nothing applies it, because every inference path loads a state dict over the
allocation. Training is the one caller that needs real starting values, so it asks here.

**This file exists because the first `base-loss` leg trained nothing.** `initial_state_dict`
returned the reference module's allocation unchanged, so every weight matrix in the model was
exactly zero at step 1. A zero matrix passes no gradient back through itself -- `dx = g W^T` is
zero -- so the only parameters that moved were the ones reachable without crossing one: the
LayerNorm scales, the IPA head weights and the output biases. Measured on the leg's own
step-1,388 checkpoint: 356 of 436 parameters had an exactly-zero RAdam `exp_avg` after 1,388
steps, and the scorer read 15.2623 A against the untrained 15.2622 A. The loss was flat to four
decimals for the whole run and every gate the campaign had was green, because each of them
compares a forward against the reference built the same zero way.

**The initializers are upstream's, by name and by formula**
(`abodybuilder3/openfold/model/primitives.py:56-114`):

* `default` -- LeCun fan-in truncated normal, `trunc_normal_init_(w, scale=1.0)`;
* `relu` -- the same at `scale=2.0`, i.e. He;
* `glorot` -- `nn.init.xavier_uniform_(w, gain=1)`;
* `normal` -- `nn.init.kaiming_normal_(w, nonlinearity="linear")`;
* `final` -- zeros, weight and bias;
* `gating` -- zero weight, bias one.

Upstream draws the truncated normal with `scipy.stats.truncnorm.rvs`, which numpy seeds and
torch does not. The draw here is `torch.nn.init.trunc_normal_` at the same std and the same
+-2 sigma truncation, so `torch.manual_seed(seed)` alone fixes every weight in the model. That
matters more than matching upstream's exact stream, which no reproduction can do anyway: both
data-parallel ranks build the model from the seed rather than broadcasting rank 0's copy, and
the step-1 master hashes agreeing is what proves it. `tests/test_abb3_init.py` holds the std
constant against `scipy.stats.truncnorm` itself where scipy is installed.

**Coverage is enforced rather than reviewed.** :func:`initialise_` fills every parameter with
NaN first and refuses to return while one is still NaN, naming it. A `Linear` added without an
`init`, a bare `nn.Parameter` nobody thought about, a module whose `reset_parameters` does not
cover all of its tensors: each of those is the defect this file was written for, and each fails
loudly instead of shipping a weight that is silently zero.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..abodybuilder3_reference import InvariantPointAttention, LayerNorm
from ..af2_reference import Linear

__all__ = ["initialise_", "SOFTPLUS_INVERSE_1", "TRUNC_NORMAL_STD_2SIGMA"]

#: `truncnorm.std(a=-2, b=2)`, the standard deviation of a unit normal truncated at +-2 sigma.
#: Upstream divides by it so that the DRAWN std is the one it asked for rather than the
#: pre-truncation one (`primitives.py:77`). Held against scipy in the tests.
TRUNC_NORMAL_STD_2SIGMA = 0.8796256610342398

#: `ipa_point_weights_init_` (`primitives.py:111-114`): the value whose softplus is 1, so the
#: point attention starts with unit head weights. Zero here would start it at softplus(0)=0.693.
SOFTPLUS_INVERSE_1 = 0.541324854612918


def _trunc_normal_(weight: torch.Tensor, scale: float) -> None:
    """`trunc_normal_init_`: fan-in scaled, truncated at +-2 sigma of the pre-truncation normal."""
    fan_in = weight.shape[-1]
    std = math.sqrt(scale / max(1, fan_in)) / TRUNC_NORMAL_STD_2SIGMA
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


def _init_linear_(module: Linear) -> None:
    name = module.init
    w, b = module.weight, module.bias
    if name == "default":
        _trunc_normal_(w, 1.0)
    elif name == "relu":
        _trunc_normal_(w, 2.0)
    elif name == "glorot":
        nn.init.xavier_uniform_(w, gain=1.0)
    elif name == "normal":
        nn.init.kaiming_normal_(w, nonlinearity="linear")
    elif name in ("final", "gating"):
        w.fill_(0.0)
    else:
        raise ValueError(f"unknown initializer {name!r}; upstream's set is default, relu, "
                         f"glorot, normal, final, gating")
    b.fill_(1.0 if name == "gating" else 0.0)


@torch.no_grad()
def initialise_(model: nn.Module, seed: int) -> nn.Module:
    """Draw every parameter of `model` in place from `seed`. Returns the model.

    Raises `RuntimeError` naming any parameter left uninitialised, which is the only way a
    silently-zero weight gets caught before it costs a training leg.
    """
    torch.manual_seed(int(seed))
    for p in model.parameters():
        p.fill_(float("nan"))

    for module in model.modules():
        if isinstance(module, Linear):
            _init_linear_(module)
        elif isinstance(module, InvariantPointAttention):
            # Upstream's `ipa_point_weights_init_` (structure_module.py:243-244).
            module.head_weights.fill_(SOFTPLUS_INVERSE_1)
        elif isinstance(module, LayerNorm):
            module.weight.fill_(1.0)
            module.bias.fill_(0.0)

    missed = [name for name, p in model.named_parameters() if torch.isnan(p).any()]
    if missed:
        raise RuntimeError(
            f"{len(missed)} parameter(s) were never initialised, first few {missed[:5]}. "
            f"Add the initializer where the parameter is declared -- a weight that stays at its "
            f"allocation is the defect that cost the first base-loss leg 1,388 steps.")
    return model

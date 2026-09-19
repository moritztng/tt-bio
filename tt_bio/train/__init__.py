"""Training for tt-bio, in four tiers of progressive disclosure.

Easy on the surface, and all of the power underneath, with the boundaries cut where the unit
of **user authorship** changes rather than where the amount of configuration does. That
distinction is the whole design: a user can name their authorship unit before they start,
while "how much configuration" is only knowable after you hit the wall.

    tier  surface                                            you own          the cut line
    0     tt-bio finetune ...                                a config file    no callables in the signature
    1     train.finetune(...) -> Run                         the objective    no `for` over steps in your code
    2     plan, batches, objectives, AdamW, Checkpointer,     the `for`        no ttnn call in your code
          Mesh, LoraConfig, trainable, attach
    3     tt_bio.autograd + train.gradcheck                  the op and       ttnn appears here
                                                             its backward

Each cut line has a one-line test in ``tests/test_train_interface.py``, and each is a test the
boundary can FAIL rather than a description of it.

Going from a LoRA fine-tune to a full pre-training run is not a drop down a tier either. It is
``train="weights"``, one argument at every tier, and it runs the SAME body: the loop, the
objective, the optimizer, the checkpointer and the data-parallel axis do not change, only what
the optimizer owns does.

Crossing down a tier is not a rewrite. ``train.recipes.source("default")`` returns the text of
the Tier-1 body, written in Tier-2 names only, and a test execs that text in a namespace
containing nothing but ``TIER2`` and checks its bytecode against the shipped recipe's. If a
recipe ever needs a private hook, that test fails and the hook becomes public or the recipe
changes.

OPT-IN AND INERT WHEN OFF. Nothing on the inference path imports this package, and the tape it
drives prunes every node when grad is off, so importing ``tt_bio`` does not change one byte of
inference behaviour. Every name below resolves lazily through ``__getattr__``, so importing
``tt_bio.train`` itself costs no ttnn import either -- which is what lets Tier 0 decide whether
a flag combination is legal before a device opens.

The adapters attach inside the shipped ``tt_bio.ops.linear``, so the training path calls the
SAME forward the inference path calls rather than a copy of it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Tier 2's public vocabulary, and the only vocabulary a Tier-1 recipe body may use. The
# escape-hatch test reads this tuple, so adding a name here is a deliberate widening of the
# contract and removing one breaks a test rather than a user's script silently.
TIER2 = (
    "plan", "batches", "Mesh", "AdamW", "Checkpointer", "LoraConfig", "trainable",
    "lora_factors_for", "weights_for", "walked_weights", "lora_factors", "lora_linear",
    "attach", "census",
    "select", "af3_lr", "to_host",
    "to_device", "objectives", "losses", "provenance", "save_adapter", "load_adapter",
    "install", "uninstall", "backward", "no_grad", "Tensor", "UnreducedGradients",
    "UNMEASURED", "launcher",
)

# name -> the submodule it lives in. A module of its own is listed as itself.
_WHERE = {
    "finetune": "loop", "Run": "loop",
    "plan": "dryrun", "Plan": "dryrun", "UNMEASURED": "dryrun",
    "CARD_DRAM_BYTES": "dryrun",
    "batches": "sharding", "Batch": "sharding",
    "Mesh": "mesh", "Axis": "mesh", "UnreducedGradients": "mesh",
    "AdamW": "optim", "af3_lr": "optim", "DISPLACEMENT_BAND": "optim",
    "LoraConfig": "lora", "LoraSite": "lora", "lora_factors": "lora",
    "lora_factors_for": "lora", "lora_linear": "lora", "census": "lora", "select": "lora",
    "attach": "lora", "trainable": "lora", "weights_for": "lora",
    "walked_weights": "lora", "Parameters": "lora",
    "Checkpointer": "checkpoint", "save_adapter": "checkpoint",
    "load_adapter": "checkpoint",
    "to_host": "tensors", "to_device": "tensors",
    "gradcheck": "checks", "GradcheckReport": "checks", "BARS": "checks",
    "Objective": "objectives",
}
# Submodules exposed as attributes. No entry here shares a name with a public FUNCTION, which
# is why `plan()` lives in `dryrun.py`, `batches()` in `sharding.py` and `gradcheck()` in
# `checks.py`: `from tt_bio.train import plan` resolves a submodule before it consults
# `__getattr__`, so a module named `plan` would hand a user the module where TIER2 promised a
# callable. Renaming the three files is a one-time cost; the shadowing would have been
# permanent.
_SUBMODULES = ("losses", "objectives", "provenance", "recipes", "mesh", "optim", "lora",
               "checkpoint", "tensors", "loop", "cli", "dryrun", "sharding", "checks",
               "catalogue", "launcher")
_FROM_AUTOGRAD = ("install", "uninstall", "installed", "is_grad_enabled", "backward",
                  "no_grad", "Tensor")

__all__ = sorted({*TIER2, *_WHERE, *_SUBMODULES, *_FROM_AUTOGRAD, "TIER2", "tier2"})

if TYPE_CHECKING:  # for editors only; never executed, so it cannot import ttnn at runtime
    from .sharding import Batch, batches
    from .checkpoint import Checkpointer, load_adapter, save_adapter
    from .checks import gradcheck
    from .lora import (LoraConfig, Parameters, attach, lora_factors, lora_factors_for,
                       lora_linear, trainable, walked_weights, weights_for)
    from .loop import Run, finetune
    from .mesh import Axis, Mesh, UnreducedGradients
    from .optim import AdamW, af3_lr
    from .dryrun import Plan, plan
    from .tensors import to_device, to_host


def __getattr__(name):
    """Resolve a public name on first use.

    Lazy because ``tt_bio.train.plan`` and the Tier-0 CLI must answer without ttnn -- the
    Tier-0 cut line is that a flag's legality is decidable before a device opens, and an eager
    re-export of ``AdamW`` would import ttnn to answer ``--help``.
    """
    where = _WHERE.get(name)
    if where is not None:
        return getattr(importlib.import_module(f".{where}", __name__), name)
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _FROM_AUTOGRAD:
        return getattr(importlib.import_module("..autograd", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__


def tier2() -> dict:
    """The Tier-2 namespace as a dict, which is what the escape-hatch test execs against.

    Here rather than in the test so the contract lives with the code it constrains. A test
    that built its own idea of the vocabulary would pass while the vocabulary drifted.
    """
    return {n: __getattr__(n) for n in TIER2}

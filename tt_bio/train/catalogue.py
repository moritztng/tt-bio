"""How a model becomes fine-tunable from Tier 0: one registration, per model.

**This registry ships empty, and that is the honest state rather than an oversight.** Tier 0
needs two things from a model that the interface cannot supply: a forward it can call with a
batch, and a featuriser that turns a path into batches. The first exists -- it is the shipped
forward, and A1's dispatch is what makes it adaptable. The second does not, and the plan
refuses to invent it: *featurisers stay per model*, because each family's cropping and MSA
handling is exactly the part that is genuinely different, and a shared data layer is the
biggest trap in this design rather than the biggest win.

So a model arrives here by registering an adapter next to its own featuriser, and until one
does, ``tt-bio finetune --model X`` refuses with the name of what is missing. Everything Tier
0 can decide without a device still works: ``--dry-run``, ``--show-recipe``,
``--list-objectives`` and every flag legality check. That is the whole of Tier 0's cut line,
and it holds with an empty registry.

Registering is one call and it is deliberately small, so the temptation to centralise the
featuriser never has a foothold::

    from tt_bio.train import catalogue

    catalogue.register("abodybuilder3", lambda path, tokens=None: (forward, Dataset(path)))

The ``dataset`` the adapter returns needs ``__len__``, ``tokens``, ``device`` and
``batch(indices) -> dict`` carrying the labels the objective row names. Four members, no base
class to inherit: a dataset is data, and the reason to keep the contract this thin is that
every family will want to satisfy it differently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

__all__ = ["register", "load", "names", "REQUIRED_DATASET_MEMBERS"]


REQUIRED_DATASET_MEMBERS = ("__len__", "tokens", "device", "batch")

_ADAPTERS: Dict[str, Callable] = {}


def register(model: str, adapter: Callable) -> Callable:
    """Register ``model``'s ``(path, tokens=None) -> (forward, dataset)`` adapter."""
    if model in _ADAPTERS and _ADAPTERS[model] is not adapter:
        raise ValueError(f"{model!r} already has a training adapter. Two adapters for one "
                         f"model name means a run cannot say which featuriser it used")
    _ADAPTERS[model] = adapter
    return adapter


def names() -> list:
    return sorted(_ADAPTERS)


def load(model: str, path: Path, *, tokens=None):
    """Resolve ``model`` to ``(forward, dataset)``, or refuse with what is missing."""
    adapter = _ADAPTERS.get(model)
    if adapter is None:
        raise NotImplementedError(
            f"no training adapter registered for {model!r}. What is missing is the "
            f"FEATURISER, not the interface: the forward is already adaptable wherever A1's "
            f"dispatch routed it, and the tiers above this call are complete. Per-model "
            f"featurisation is deliberately not generalised -- register one with "
            f"tt_bio.train.catalogue.register({model!r}, adapter). "
            f"Registered today: {names() or 'none'}")
    forward, dataset = adapter(Path(path), tokens=tokens)
    missing = [m for m in REQUIRED_DATASET_MEMBERS if not hasattr(dataset, m)]
    if missing:
        raise TypeError(f"{model!r}'s dataset is missing {missing}; the contract is "
                        f"{list(REQUIRED_DATASET_MEMBERS)}")
    return forward, dataset

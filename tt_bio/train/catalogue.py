"""How a model becomes fine-tunable from Tier 0: one registration, per model.

Tier 0 needs two things from a model that the interface cannot supply: a forward it can call
with a batch, and a featuriser that turns a path into batches. *Featurisers stay per model*,
because each family's cropping and MSA handling is exactly the part that is genuinely
different, and a shared data layer is the biggest trap in this design rather than the biggest
win. OpenFold3 is the one shipped entry (``SHIPPED``, imported on first use, so naming the
model is enough); any other name refuses with what is missing, and everything Tier 0 decides
without a device still works for it: ``--dry-run``, ``--show-recipe``, ``--list-objectives``
and every flag legality check.

Registering is one call and it is deliberately small, so the temptation to centralise the
featuriser never has a foothold::

    from tt_bio.train import catalogue

    catalogue.register("abodybuilder3", lambda path, tokens=None: (forward, Dataset(path)))

The ``dataset`` the adapter returns needs ``__len__``, ``tokens``, ``device`` and
``batch(indices) -> dict`` carrying the labels the objective row names. Four members, no base
class to inherit: a dataset is data, and the reason to keep the contract this thin is that
every family will want to satisfy it differently.

``device`` must be resolved LAZILY, on first use, and not in the constructor. Data parallelism
is one process per chip, so the process that spawns the ranks has to reach the launcher holding
no card; a dataset that opens a device while being built takes a chip in the driver and the
launcher refuses to start. The contract check below looks the four members up without calling
them for the same reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

__all__ = ["register", "load", "names", "REQUIRED_DATASET_MEMBERS"]


REQUIRED_DATASET_MEMBERS = ("__len__", "tokens", "device", "batch")

_ADAPTERS: Dict[str, Callable] = {}

#: Shipped adapters, imported on first use so naming a model is enough to train it. Importing
#: one pulls in its model's featuriser, which `tt-bio train --help` and a dry run must not pay.
SHIPPED = {"openfold3": "tt_bio.train.openfold3"}


def register(model: str, adapter: Callable) -> Callable:
    """Register ``model``'s ``(path, tokens=None) -> (forward, dataset)`` adapter."""
    if model in _ADAPTERS and _ADAPTERS[model] is not adapter:
        raise ValueError(f"{model!r} already has a training adapter. Two adapters for one "
                         f"model name means a run cannot say which featuriser it used")
    _ADAPTERS[model] = adapter
    return adapter


def names() -> list:
    return sorted({*_ADAPTERS, *SHIPPED})


def load(model: str, path: Path, *, tokens=None):
    """Resolve ``model`` to ``(forward, dataset)``, or refuse with what is missing."""
    if model not in _ADAPTERS and model in SHIPPED:
        import importlib
        importlib.import_module(SHIPPED[model])
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
    # Looked up on the CLASS, then in the instance's own dict -- never with `hasattr`, which
    # CALLS a property to find out whether it is there. `device` is the one that matters: a
    # data-parallel run is one process per chip, so a dataset resolves its device lazily and
    # the driver must reach the launcher holding no card. `hasattr(dataset, "device")` opened
    # one, in the driver, from inside the check that exists to validate the contract.
    missing = [m for m in REQUIRED_DATASET_MEMBERS
               if not (hasattr(type(dataset), m) or m in vars(dataset))]
    if missing:
        raise TypeError(f"{model!r}'s dataset is missing {missing}; the contract is "
                        f"{list(REQUIRED_DATASET_MEMBERS)}")
    return forward, dataset

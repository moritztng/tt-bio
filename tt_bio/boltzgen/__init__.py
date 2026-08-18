"""BoltzGen vendored into tt-bio, stripped to Tenstorrent-only inference.

User-facing entry points live in :mod:`tt_bio.boltzgen.cli` (CLI) and
:mod:`tt_bio.boltzgen.adapter` (the ttnn-module adapters + checkpoint
loader). Everything else is the vendored BoltzGen source, with training-only
code (validators, optimizers, training-time losses, training data filters/
samplers) removed and PyTorch Lightning + Hydra dependencies eliminated.

The shipping BoltzGen checkpoints pickle ``hyper_parameters`` that reference
``boltzgen.X`` classes. Pickle imports ``boltzgen`` before resolving anything
under it, so we keep a bare module alias here. Resolution of deleted
*training-only* class names is scoped to ``adapter._legacy_pickle_compat``.
"""
import sys as _sys

_sys.modules.setdefault("boltzgen", _sys.modules[__name__])

# The adapter pulls in tt_bio.tenstorrent (ttnn) at module scope. Import it lazily
# so importing anything else under this package (e.g. tt_bio.data.mol ->
# boltzgen.model.geometry) works on hosts without the Tenstorrent SDK (issue #6).
def __getattr__(name):
    if name in __all__:
        from tt_bio.boltzgen import adapter

        return getattr(adapter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "load_boltz_checkpoint",
    "TTPairformerNoSeqModule",
    "TTScoreModelAdapter",
]

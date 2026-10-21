"""The interface the RFD4 device implementation plugs into.

The host path (spec -> features -> CIF) and the model are separate rows, so this names what
one owes the other and nothing else. An engine takes a feature batch and returns four
tensors; ``write_design`` needs exactly those and reads nothing else off the model:

    coords_nm        (designs, tokens, 37, 3)   nanometres, atom37 slots
    residue_type     (designs, tokens)          vocabulary indices
    atom_mask        (designs, tokens, 37)      which atom37 slots are real
    insertion_logits (designs, tokens, tokens)  unindexed-motif insertion mapping

Until the ttnn port lands, ``get_engine`` returns the stub, which raises with the flags the
user passed rather than pretending to design something.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class DesignOutput:
    """What an engine returns. Attribute names are upstream's, so write_cif takes it as is."""

    coords_nm: object
    residue_type: object
    atom_mask: object
    insertion_logits: object


@runtime_checkable
class Engine(Protocol):
    """Generate designs from one featurized example."""

    def design(self, features: dict, *, num_designs: int, nsteps: int, seed: int) -> DesignOutput:
        ...


class NotPortedEngine:
    """Stands in until the device implementation lands, and says so precisely."""

    def design(self, features: dict, *, num_designs: int, nsteps: int, seed: int) -> DesignOutput:
        tokens = features["padding_mask"].shape[-1]
        raise NotImplementedError(
            f"The RFD4 model is not on Tenstorrent yet: {num_designs} design(s) of {tokens} "
            f"tokens at {nsteps} steps, seed {seed}, were prepared but not generated. "
            "The inputs are written out and are what the model will read."
        )


def get_engine(name: str = "auto") -> Engine:
    """The engine to design with. Only the stub exists today."""
    if name not in ("auto", "stub"):
        raise ValueError(f"unknown RFD4 engine {name!r}")
    return NotPortedEngine()

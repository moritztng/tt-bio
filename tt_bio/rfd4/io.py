"""RFD4's host path: design spec in, features out, annotated CIF back.

Three steps, all of them upstream's own code under ``tt_bio._vendor.rfd4_prep``:

    spec (YAML/JSON) --load_spec--> examples
    example --featurize--> (features, atom_array)
    engine output --write_design--> <example_id>_generated.cif

A spec example names input structures, a contig and conditions; building it into an
annotated AtomArray is ``build_annotated_atom_array_from_condition_spec``, and
``prepare_cif`` featurizes an annotated CIF. The intermediate CIF is written out because
it is also what the user wants to see: it is the input to the model in the form the model
reads it, and upstream writes the same sidecar.

Featurization is seeded. ``RealizeExpandableSegments`` samples each expandable segment's
length, so the same spec at two seeds is two different designs and two different token
counts. The seed is part of the design's identity, not a knob.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The token length a design is padded to is decided by the engine, not here;
#: prepare_cif pads to the example's own length and fills absent schema keys.
__all__ = ["DesignInput", "featurize", "load_spec", "prepare_examples", "write_design"]


@dataclass(frozen=True)
class DesignInput:
    """One design: its id, the annotated CIF it featurizes from, and where it came from."""

    example_id: str
    cif: Path
    spec: dict[str, Any] | None = None


def load_spec(path: str | Path) -> dict[str, dict]:
    """``{example_id: example}`` from a design spec file, global_args folded in.

    Raises ``NotImplementedError`` on RFD4's legacy contig-JSON dialect, the same way
    upstream's ConditionSpecDataset does, rather than half-reading it.
    """
    from tt_bio._vendor.rfd4_prep._cif.inference_utils_condition_spec import (
        is_legacy_dialect, load_spec_file, split_global_args,
    )

    path = Path(path).expanduser().resolve()
    spec = load_spec_file(path)
    if is_legacy_dialect(spec):
        raise NotImplementedError(
            f"{path} uses RFD4's legacy contig-JSON dialect. Use a condition-spec file "
            "(the `input`/`contig`/`conditions` form)."
        )
    return split_global_args(spec)


def prepare_examples(spec_path: str | Path, out_dir: str | Path, *,
                     only: list[str] | None = None, validate: bool = True) -> list[DesignInput]:
    """Build every example in a spec into an annotated CIF under ``out_dir``.

    The CIFs are the model's actual inputs, kept rather than hidden in a temp dir: a design
    that comes out wrong is usually a spec that selected the wrong atoms, and that is
    visible in this file and nowhere else.
    """
    from tt_bio._vendor.rfd4_prep._cif.inference_utils_condition_spec import write_condition_spec_cif

    spec_path = Path(spec_path).expanduser().resolve()
    examples = load_spec(spec_path)
    if only:
        missing = sorted(set(only) - set(examples))
        if missing:
            raise ValueError(f"no such design(s) in {spec_path.name}: {missing}")
        examples = {k: v for k, v in examples.items() if k in only}

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for example_id, example in examples.items():
        cif = out_dir / f"{example_id}.cif"
        write_condition_spec_cif(example_id, example, cif,
                                 spec_dir=spec_path.parent, validate=validate)
        out.append(DesignInput(example_id, cif, example))
    return out


def featurize(cif: str | Path, *, seed: int, device=None, path_length_cutoff: int = 64):
    """``(features, atom_array)`` for one annotated CIF, as the model takes them.

    This is upstream's ``prepare_cif``. Features are batched (leading axis 1), padded to
    the example's token count, with every absent schema key filled by its default, so the
    dict's key set and shapes do not depend on which conditions the spec used.
    """
    import torch

    from tt_bio._vendor.rfd4_prep.processing import prepare_cif

    return prepare_cif(Path(cif).expanduser(), seed=seed,
                       device=device if device is not None else torch.device("cpu"),
                       path_length_cutoff=path_length_cutoff)


def write_design(output, features: dict, atom_array, path: str | Path, *, row: int = 0) -> Path:
    """Write one design to an annotated CIF, upstream's ``write_cif``.

    Coordinates come back in nanometres and are converted, the design is realigned onto
    the input's frame, and the file is published by link into place, so an existing path
    is never overwritten and a reader never sees a half-written file.
    """
    from tt_bio._vendor.rfd4_prep.processing import write_cif

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return Path(write_cif(output, features, atom_array, path, row=row))

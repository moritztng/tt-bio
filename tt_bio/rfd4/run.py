"""One call from a design spec to designs on disk.

``tt-bio design --model rfd4`` and :func:`design` are the same path: the CLI parses flags and
calls this, so anything the CLI can do is reachable from Python with the same arguments and
the same defaults.

    from tt_bio import rfd4

    paths = rfd4.design("binder.yaml", "designs/", num_designs=4, seed=0)

The engine is loaded once per call. Designing against several specs, or sweeping seeds, is
cheaper with the engine kept alive, which is what ``engine=`` is for:

    engine = rfd4.DesignEngine.from_pretrained()
    for seed in range(8):
        rfd4.design("binder.yaml", f"designs/seed{seed}", seed=seed, engine=engine)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

__all__ = ["design"]


def design(spec: str | Path, out_dir: str | Path = "designs", *, num_designs: int = 1,
           recipe: str | None = None, nsteps: int | None = None, seed: int = 0,
           only: Sequence[str] | None = None,
           validate: bool = True, annotated_cif: bool = False, engine: Any = None,
           on_design: Callable[[str, int, Path], None] | None = None,
           on_write: Callable[[Path], None] | None = None) -> list[Path]:
    """Design from ``spec``, write one annotated CIF per design, return their paths.

    ``spec`` is an RFD4 condition spec (YAML or JSON), or an annotated CIF when
    ``annotated_cif=True``. ``only`` selects design ids from the spec; ``num_designs`` is how
    many designs each of them produces. ``nsteps`` defaults to the engine recipe's step count,
    and a given ``seed`` reproduces a run exactly, featurization included.

    ``on_design(example_id, tokens, cif)`` is called as each design's input is featurized and
    ``on_write(path)`` as each output lands; the CLI prints its progress from both.
    """
    from tt_bio.rfd4.engine import get_engine
    from tt_bio.rfd4.io import DesignInput, featurize, prepare_examples, write_design

    if recipe is not None:
        from tt_bio.rfd4.sampler import recycle_strategy

        if engine is not None:
            raise ValueError("an engine already carries its recipe; pass one or the other")
        recycle_strategy(recipe)  # an unknown name fails here, before any design is built
    spec = Path(spec).expanduser()
    out_dir = Path(out_dir).expanduser()
    if annotated_cif:
        if only:
            raise ValueError("`only` selects designs from a spec file; an annotated CIF is "
                             "one design's input already")
        inputs = [DesignInput(spec.stem, spec)]
    else:
        inputs = prepare_examples(spec, out_dir / "inputs", only=list(only) if only else None,
                                  validate=validate)

    engine = engine if engine is not None else get_engine(
        **({"recipe": recipe} if recipe is not None else {}))
    if nsteps is None:
        nsteps = getattr(getattr(engine, "spec", None), "nsteps", 60)

    written: list[Path] = []
    for design_input in inputs:
        features, atom_array = featurize(design_input.cif, seed=seed)
        if on_design is not None:
            on_design(design_input.example_id, int(features["padding_mask"].shape[-1]),
                      design_input.cif)
        output = engine.design(features, num_designs=num_designs, nsteps=nsteps, seed=seed)
        for row in range(num_designs):
            suffix = "" if num_designs == 1 else f"_{row}"
            path = write_design(output, features, atom_array,
                                out_dir / f"{design_input.example_id}{suffix}_generated.cif",
                                row=row)
            if on_write is not None:
                on_write(path)
            written.append(path)
    return written

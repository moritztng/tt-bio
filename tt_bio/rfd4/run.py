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

__all__ = ["BACKENDS", "check_backend", "design"]


#: Upstream's `inference_backend`. `native` is the model itself, which is what tt-bio ports.
#: `bioir` runs the whole workflow through NVIDIA's BioNeMo-IR with a CUDA-graph-captured trunk, so
#: it is a property of the GPU it was written for, not a backend a port can offer.
BACKENDS = {
    "native": None,
    "bioir": "inference_backend=bioir is NVIDIA BioNeMo-IR, which captures the trunk as a CUDA "
             "graph and runs only on NVIDIA GPUs. tt-bio runs RFD4 natively on Tenstorrent; use "
             "backend='native' (the default). The same designs come out either way.",
}


def check_backend(backend: str) -> None:
    """Refuse a backend tt-bio does not run, with the reason. Shared by the local and fleet paths."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown inference backend {backend!r}; have {sorted(BACKENDS)}")
    if BACKENDS[backend]:
        raise ValueError(BACKENDS[backend])


def design(spec: str | Path, out_dir: str | Path = "designs", *, num_designs: int = 1,
           backend: str = "native", recipe: str | None = None, adapter: str | Path | None = None,
           tsr_k: float | None = None, nsteps: int | None = None, seed: int = 0,
           only: Sequence[str] | None = None,
           validate: bool = True, annotated_cif: bool = False, engine: Any = None,
           sampling: dict | None = None, transforms: Sequence[Callable] = (), guidance=None,
           save_trajectory_every: int = 0,
           on_design: Callable[[str, int, Path], None] | None = None,
           on_write: Callable[[Path], None] | None = None) -> list[Path]:
    """Design from ``spec``, write one annotated CIF per design, return their paths.

    ``spec`` is an RFD4 condition spec (YAML or JSON), or an annotated CIF when
    ``annotated_cif=True``. ``only`` selects design ids from the spec; ``num_designs`` is how
    many designs each of them produces. ``nsteps`` defaults to the engine recipe's step count,
    and a given ``seed`` reproduces a run exactly, featurization included.

    ``sampling`` holds upstream's partial-diffusion and inverse-folding arguments
    (``tt_bio.rfd4.sampler.resolve_partial_start``), ``transforms`` the per-replicate sampling
    transforms and ``guidance`` a metric-gradient callable. ``save_trajectory_every`` writes a
    CA-only multi-model CIF beside each design.

    ``on_design(example_id, tokens, cif)`` is called as each design's input is featurized and
    ``on_write(path)`` as each output lands; the CLI prints its progress from both.
    """
    from tt_bio.rfd4.engine import get_engine
    from tt_bio.rfd4.io import featurize, resolve_inputs, start_writer, write_designs

    check_backend(backend)
    if engine is not None and (recipe is not None or adapter is not None or tsr_k is not None):
        raise ValueError("an engine already carries its recipe and adapter, and its sequence "
                         "temperature; pass one or the other")
    sampling = dict(sampling or {})
    if recipe is not None or sampling:
        from tt_bio.rfd4.sampler import recycle_strategy, resolve_partial_start

        # An unknown recipe, and upstream's mutual exclusions between the sampling arguments,
        # are host arithmetic: settle them before a card comes up and a bundle is read.
        strategy = (getattr(engine, "spec", None) if engine is not None else None) \
            or recycle_strategy(recipe if recipe is not None else "thinking_medium")
        if sampling:
            resolve_partial_start(strategy, **sampling)
    if adapter is not None and not Path(adapter).expanduser().is_dir():
        raise ValueError(f"no LoRA adapter directory at {adapter}")
    out_dir = Path(out_dir).expanduser()
    inputs = resolve_inputs(spec, out_dir / "inputs", only=only, validate=validate,
                            annotated_cif=annotated_cif)

    loaded = {k: v for k, v in (("recipe", recipe), ("adapter", adapter), ("tsr_k", tsr_k))
              if v is not None}
    engine = engine if engine is not None else get_engine(**loaded)
    if nsteps is None:
        nsteps = getattr(getattr(engine, "spec", None), "nsteps", 60)

    written: list[Path] = []
    for design_input in inputs:
        features, atom_array = featurize(design_input.cif, seed=seed)
        if on_design is not None:
            on_design(design_input.example_id, int(features["padding_mask"].shape[-1]),
                      design_input.cif)
        # Only what was asked for: an engine that offers neither steering nor trajectories is
        # still a valid engine for a default run.
        steering = {k: v for k, v in (("transforms", transforms), ("guidance", guidance),
                                      ("save_trajectory_every", save_trajectory_every)) if v}
        start_writer(num_designs)  # its start-up runs while the card designs
        output = engine.design(features, num_designs=num_designs, nsteps=nsteps, seed=seed,
                               **steering, **sampling)
        suffixes = [""] if num_designs == 1 else [f"_{row}" for row in range(num_designs)]
        paths = write_designs(output, features, atom_array,
                              [out_dir / f"{design_input.example_id}{x}_generated.cif" for x in suffixes])
        for row, (suffix, path) in enumerate(zip(suffixes, paths)):
            if on_write is not None:
                on_write(path)
            written.append(path)
            if getattr(output, "trajectory", None):
                from tt_bio.rfd4.trajectory import write_trajectory

                frames = write_trajectory(
                    output.trajectory, features,
                    out_dir / f"{design_input.example_id}{suffix}_trajectory.cif", row=row)
                if on_write is not None:
                    on_write(frames)
    return written

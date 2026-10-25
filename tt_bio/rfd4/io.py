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
import os
import sys
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The token length a design is padded to is decided by the engine, not here;
#: prepare_cif pads to the example's own length and fills absent schema keys.
__all__ = ["DesignInput", "featurize", "load_spec", "prepare_examples", "resolve_inputs",
           "start_writer", "write_design", "write_designs",
           "writer_count", "writer_pool"]


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


def resolve_inputs(spec: str | Path, out_dir: str | Path, *, only=None, validate: bool = True,
                   annotated_cif: bool = False) -> list[DesignInput]:
    """The designs ``spec`` asks for, each as an annotated CIF on disk.

    ``annotated_cif`` means ``spec`` IS one design's prepared input, so nothing is built. Both the
    local and the fleet path come through here, so a spec is read and validated the same way
    whichever one runs it.
    """
    spec = Path(spec).expanduser()
    if not annotated_cif:
        return prepare_examples(spec, out_dir, only=list(only) if only else None, validate=validate)
    if only:
        raise ValueError("`only` selects designs from a spec file; an annotated CIF is "
                         "one design's input already")
    return [DesignInput(spec.stem, spec)]


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

    `features` is one example and `output` is `num_designs` of it, so the writer's row index
    runs past the features' single row. Upstream repeats the features per design before it
    writes; we hand the writer the same view instead of carrying `num_designs` copies of every
    feature through the engine.
    """
    from tt_bio._vendor.rfd4_prep.processing import write_cif
    from tt_bio.rfd4.network import _broadcast_batch

    # `features` is the one input every design shares (a leading axis of 1), and write_cif
    # indexes it by `row` like the output, so it gets the output's design axis first.
    features = _broadcast_batch(features, output.coords_nm.shape[0])
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return Path(write_cif(output, features, atom_array, path, row=row))


#: One design of an engine output: the four fields `write_cif` reads, cut to batch 1.
_Row = namedtuple("_Row", "coords_nm residue_type atom_mask insertion_logits")

#: Most writer processes a batch gets. A write is ~0.2 s, so past this the start-up and the last
#: straggler eat what another process would save, and a host running one engine per chip
#: multiplies whatever this is by the number of engines.
MAX_WRITERS = 8
#: Memory the writer's forkserver holds (Python, torch, biotite's decoded CCD) and memory one
#: writer adds beyond the pages it shares with the server, rounded up from pc at 224 tokens
#: (perf/rfd4/gap224mem: ~1.66 GB and <=70 MB).
SERVER_MIB = 1800
WRITER_MIB = 128
_POOL = None


def _available_mib() -> int:
    try:
        with open("/proc/meminfo") as f:
            return next(int(line.split()[1]) // 1024 for line in f if line.startswith("MemAvailable:"))
    except (OSError, StopIteration):
        return 0


def writer_count(n_designs: int, available_mib: int | None = None) -> int:
    """How many processes write a batch of `n_designs`: one per design, at most `MAX_WRITERS` and
    one per CPU, and only as many as fit, with the server, in half of the memory available now.
    1 means write in this process."""
    if available_mib is None:
        available_mib = _available_mib()
    fit = (available_mib // 2 - SERVER_MIB) // WRITER_MIB
    return max(1, min(n_designs, MAX_WRITERS, os.cpu_count() or 1, fit))


@contextmanager
def _main_hidden():
    """Start writer processes without re-running the caller's script: a spawned process executes
    `__main__` as `__mp_main__` before its first task, and a harness or the CLI imports the
    device stack at the top. The writer needs none of it."""
    main = sys.modules["__main__"]
    saved = {k: main.__dict__[k] for k in ("__file__", "__spec__") if k in main.__dict__}
    main.__dict__.pop("__file__", None)
    main.__spec__ = None
    try:
        yield
    finally:
        main.__dict__.update(saved)


#: What a writer needs loaded before its first write, single-threaded (see the module).
_PRELOAD = "tt_bio.rfd4._writer_preload"


def _context():
    import multiprocessing

    ctx = multiprocessing.get_context("forkserver")
    ctx.set_forkserver_preload([_PRELOAD])
    return ctx


def start_writer(n_designs: int) -> None:
    """Start the server `write_designs` forks its writers from, if a batch of `n_designs` gets any,
    without waiting for it.

    The server loads what a write needs (`tt_bio.rfd4._writer_preload`, ~3 s) once and writers
    fork from it, so they hold none of this process's device handles or threads and share its
    memory. Call this before the card runs and the load hides behind the design.
    """
    from multiprocessing import forkserver

    if writer_count(n_designs) > 1:
        _context()
        with _main_hidden():
            forkserver.ensure_running()


def writer_pool(n_designs: int):
    """The processes a batch of `n_designs` is written on, or None to write it in this process.

    Started on the first batch and kept, so later batches pay neither a fork (~20 ms a writer)
    nor a writer's first, colder design. A larger batch, or a writer that died, replaces it.

    Each writer imports `_PRELOAD` before its first task. That is free when the forkserver loaded
    it, and it is what keeps the writer single-threaded when the forkserver was already running
    before `_context` named the preload: the preload is then silently skipped, and a writer that
    imported torch on its own would start a BLAS/OpenMP pool per process.
    """
    import importlib
    from concurrent.futures import ProcessPoolExecutor

    global _POOL
    n = writer_count(n_designs)
    if n == 1:
        return None
    if _POOL is None or _POOL._broken or _POOL._max_workers < n:
        if _POOL is not None:
            _POOL.shutdown(wait=False)
        _POOL = ProcessPoolExecutor(n, mp_context=_context(), initializer=importlib.import_module,
                                    initargs=(_PRELOAD,))  # writers start as tasks arrive
    return _POOL


def write_designs(output, features: dict, atom_array, paths) -> list[Path]:
    """`write_design` for rows 0..len(paths)-1 of `output`, several at a time.

    A write is ~0.2 s of single-threaded Python at 224 tokens and starts only once the card has
    finished, so a batch written one design after another adds that per design to the wall.
    The batch goes to `writer_pool`. Each writer gets its own row (copied, so a view does not ship
    the whole batch) and the padding mask, which is all `write_cif` reads from `features`. The
    files are identical to a serial write.
    """
    paths = list(paths)
    pool = writer_pool(len(paths))
    if pool is None:
        return [write_design(output, features, atom_array, p, row=k) for k, p in enumerate(paths)]
    mask = {"padding_mask": features["padding_mask"].clone()}
    with _main_hidden():  # writers start inside submit
        jobs = [pool.submit(write_design, _Row(*(getattr(output, f)[k:k + 1].detach().cpu().clone()
                                                  for f in _Row._fields)), mask, atom_array, p)
                for k, p in enumerate(paths)]
    return [j.result() for j in jobs]

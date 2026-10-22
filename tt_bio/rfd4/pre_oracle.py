"""Pre-oracle transforms: one design, several questions for the oracle.

Upstream's `PreOracleTransform` rewrites a design before the oracle call, so one design row can
answer more than one question. `OffTargetSubstitutionTransform` is the case that motivates it:
the designed chain is replanted against each listed off-target, and the resulting scores are a
specificity assay rather than another copy of the same affinity number.

This is host-side text work on the candidate, in front of `Oracle.fold_batch`, so it costs the
oracle one extra fold per off-target and nothing else. Metrics for a variant come back under
`<transform>.<variant>.<key>`, keeping the untransformed design's keys where the reward model
already reads them.
"""
from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from tt_bio.rfd4.oracle import OracleRequest


@runtime_checkable
class PreOracleTransform(Protocol):
    """Fan one request out into named variants of itself."""

    name: str

    def expand(self, request: OracleRequest) -> list[tuple[str, OracleRequest]]:
        """`[(variant, request)]`. An empty list drops the design from this transform."""
        ...


def expand(requests: Sequence[OracleRequest],
           transforms: Sequence[PreOracleTransform]) -> tuple[list[OracleRequest], list[tuple[int, str | None]]]:
    """The jobs to fold, plus where each one's metrics belong.

    The untransformed request is always first in the batch and keeps its own keys; each variant
    follows it. The returned routing list is `(original index, metric prefix)` per job, in the
    batch's order, which is what `collect` folds back together.
    """
    jobs: list[OracleRequest] = []
    routing: list[tuple[int, str | None]] = []
    for index, request in enumerate(requests):
        jobs.append(request)
        routing.append((index, None))
        for transform in transforms:
            for variant, job in transform.expand(request):
                jobs.append(job)
                routing.append((index, f"{transform.name}.{variant}"))
    return jobs, routing


def collect(results: Sequence[dict], routing: Sequence[tuple[int, str | None]],
            n_requests: int) -> list[dict]:
    """One metric dict per original request, variants namespaced under their transform."""
    out: list[dict] = [{} for _ in range(n_requests)]
    for metrics, (index, prefix) in zip(results, routing):
        if prefix is None:
            out[index].update(metrics)
        else:
            out[index].update({f"{prefix}.{key}": value for key, value in metrics.items()})
    return out


@dataclass
class OffTargetSubstitution:
    """Replant the designed chain against each off-target: upstream's specificity assay.

    `off_targets` maps a name to the structure text of that target. The chain of interest is
    taken from the design, everything else is dropped, and the off-target's chains are relabelled
    so they cannot collide with it.
    """

    off_targets: dict[str, str]
    name: str = "off_target"

    def expand(self, request: OracleRequest) -> list[tuple[str, OracleRequest]]:
        if request.chain_of_interest is None:
            raise ValueError("an off-target substitution needs the design's chain_of_interest")
        binder = _chains(request.structure, keep=[request.chain_of_interest])
        out = []
        for variant, structure in self.off_targets.items():
            target = _chains(structure, drop=[request.chain_of_interest])
            if not len(target):
                raise ValueError(f"off-target {variant!r} has no chain besides "
                                 f"{request.chain_of_interest!r}")
            complexed = _relabel(binder, target, keep=request.chain_of_interest)
            out.append((variant, OracleRequest(
                structure=_to_text(complexed),
                example_id=f"{request.example_id}__{self.name}_{variant}",
                chain_of_interest=request.chain_of_interest,
                out_dir=request.out_dir)))
        return out


def _read(structure_text: str):
    import biotite.structure.io.pdb as pdb

    from tt_bio.af2ig import to_pdb_text

    return pdb.PDBFile.read(io.StringIO(to_pdb_text(structure_text))).get_structure(model=1)


def _chains(structure_text: str, *, keep=None, drop=None):
    atoms = _read(structure_text)
    ids = np.asarray(atoms.chain_id)
    mask = np.isin(ids, keep) if keep is not None else ~np.isin(ids, drop)
    selected = atoms[mask]
    if keep is not None and not len(selected):
        raise ValueError(f"chain(s) {keep} are not in the structure")
    return selected


def _relabel(binder, target, *, keep: str):
    """Chain ids the oracle can tell apart: the designed chain keeps its letter, the off-target
    takes the next free ones in its own order."""
    free = (c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c != keep)
    mapping = {old: next(free) for old in dict.fromkeys(np.asarray(target.chain_id))}
    relabelled = target.copy()
    relabelled.chain_id = np.array([mapping[c] for c in np.asarray(target.chain_id)])
    return binder + relabelled


def _to_text(atoms) -> str:
    import biotite.structure.io.pdb as pdb

    out = io.StringIO()
    handle = pdb.PDBFile()
    handle.set_structure(atoms)
    handle.write(out)
    return out.getvalue()

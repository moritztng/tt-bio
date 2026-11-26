"""Receptor + ligand to the graphs DiffDock-L consumes, one complex at a time.

Reproduces `utils.inference_utils.InferenceDataset` and the input half of
`inference.py`: the four-column screening CSV, the per-complex failure
isolation, the two graph builds (score and confidence), and the centring both
models expect. The ESM-2 embedding is not computed here; it comes from tt-bio's
own ESM-2 on the card (D-dfd-4) and is passed in.

Differences from upstream, all recorded in state/dfd/DECISIONS.md:

* mmCIF receptors as well as PDB (D-dfd-input-2).
* A conformer seed, so a run is reproducible; at the default there is none and
  RDKit behaves as upstream (D-dfd-input-4).
* A single-complex call raises; only a batch isolates failures, and names the
  failures instead of printing a count (FEATURE-PLAN conflict 1).
* A sequence-only receptor is folded with tt-bio's esmfold2, not upstream's
  esmfold_v1 (D-dfd-input-5).
* `conformer_timeout_s` bounds ETKDG so one runaway ligand cannot stop a screen;
  at the default there is no bound and the behaviour is upstream's
  (D-dfd-input-7).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch

from .ligand import build_ligand_graph, read_ligand
from .receptor import build_receptor_graph
from .structure import parse_receptor

#: The columns upstream's screening CSV carries, in upstream's order.
CSV_COLUMNS = ("complex_name", "protein_path", "protein_sequence", "ligand_description")

#: Graph settings of the released v1.1 checkpoints, from their
#: `model_parameters.yml`. The score model sees residues, the confidence model
#: builds its own all-atom graph (`use_original_model_cache: false`).
SCORE_GRAPH = dict(c_alpha_max_neighbors=24, all_atoms=False, atom_max_neighbors=8)
CONFIDENCE_GRAPH = dict(c_alpha_max_neighbors=24, all_atoms=True, atom_max_neighbors=8)


@dataclass
class ComplexSpec:
    """One row of work: a name, a receptor, a ligand."""

    name: str
    protein_path: str | None = None
    protein_sequence: str | None = None
    ligand_description: str = ""


@dataclass
class ComplexGraph:
    """The graphs for one complex, plus what the sampler and writer need.

    `receptor` and `ligand` are the dicts the two builders return.
    `original_center` is the receptor centroid that was subtracted, which the
    pose writer adds back. `mol` keeps its hydrogens; the graph does not.
    """

    name: str
    receptor: dict
    ligand: dict
    mol: object
    original_center: torch.Tensor
    chain_sequences: list[str] = field(default_factory=list)


def build_complex_graph(
    spec: ComplexSpec,
    *,
    all_atoms: bool = False,
    lm_embeddings: list | None = None,
    seed: int | None = None,
    receptor_path: str | Path | None = None,
    conformer_timeout_s: float | None = None,
) -> ComplexGraph:
    """Build one complex's graph. Raises on anything it cannot parse.

    `receptor_path` overrides `spec.protein_path`, which is how a folded
    sequence-only receptor is fed back in.
    """
    path = receptor_path if receptor_path is not None else spec.protein_path
    if path is None:
        raise ValueError(
            f"{spec.name}: no receptor structure; fold the sequence first "
            "(tt_bio.diffdock.input.fold_sequence_receptors)")

    mol = read_ligand(spec.ligand_description, seed=seed,
                      conformer_timeout_s=conformer_timeout_s)
    ligand = build_ligand_graph(mol, remove_hs=True)

    receptor_struct = parse_receptor(path)
    settings = CONFIDENCE_GRAPH if all_atoms else SCORE_GRAPH
    receptor = build_receptor_graph(receptor_struct, lm_embeddings=lm_embeddings, **settings)

    center = receptor["pos"].mean(dim=0, keepdim=True)
    receptor["pos"] = receptor["pos"] - center
    if "atom" in receptor:
        receptor["atom"]["pos"] = receptor["atom"]["pos"] - center
    ligand["pos"] = ligand["pos"] - ligand["pos"].mean(dim=0, keepdim=True)

    return ComplexGraph(
        name=spec.name,
        receptor=receptor,
        ligand=ligand,
        mol=mol,
        original_center=center,
        chain_sequences=list(receptor["sequence"]),
    )


def read_complex_csv(path: str | Path) -> list[ComplexSpec]:
    """Read upstream's four-column screening CSV.

    Upstream reads it with pandas, which turns an empty cell into NaN and then
    maps NaN to None (`set_nones`). The same happens here without pandas: an
    empty or whitespace-only cell, or the literal `nan` pandas would have
    produced, is None. A row with no name gets `complex_<i>`, as upstream.
    """
    path = Path(path)
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no rows")
    missing = [c for c in CSV_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing the column(s) {', '.join(missing)}")

    def clean(value):
        value = (value or "").strip()
        return None if value == "" or value.lower() == "nan" else value

    specs = []
    for i, row in enumerate(rows):
        name = clean(row["complex_name"]) or f"complex_{i}"
        specs.append(ComplexSpec(
            name=name,
            protein_path=clean(row["protein_path"]),
            protein_sequence=clean(row["protein_sequence"]),
            ligand_description=clean(row["ligand_description"]) or "",
        ))
    return specs


@dataclass
class BatchResult:
    """What a screening run produced, and what it could not."""

    graphs: list[ComplexGraph] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def skipped(self) -> int:
        return len(self.failures)

    def write_failures_csv(self, path: str | Path) -> None:
        """Name every failed complex and its error, one row each."""
        with Path(path).open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(("complex_name", "error"))
            writer.writerows(self.failures)


def build_batch(
    specs: list[ComplexSpec],
    *,
    all_atoms: bool = False,
    seed: int | None = None,
    embeddings_for: Callable[[ComplexSpec], list] | None = None,
    receptor_paths: dict[str, str] | None = None,
    conformer_timeout_s: float | None = None,
) -> BatchResult:
    """Build every complex in a batch, isolating the ones that fail.

    Upstream swallows the exception, prints it and counts it. tt-bio keeps the
    isolation, which a 10,000-ligand screen needs, and keeps the reason: each
    failure comes back as `(name, error)` for `failures.csv`.
    """
    result = BatchResult()
    for spec in specs:
        try:
            result.graphs.append(build_complex_graph(
                spec,
                all_atoms=all_atoms,
                seed=seed,
                lm_embeddings=embeddings_for(spec) if embeddings_for else None,
                receptor_path=(receptor_paths or {}).get(spec.name),
                conformer_timeout_s=conformer_timeout_s,
            ))
        except Exception as exc:  # noqa: BLE001 - one bad row must not end the screen
            result.failures.append((spec.name, f"{type(exc).__name__}: {exc}"))
    return result


def iter_sequence_receptors(specs: list[ComplexSpec]) -> Iterator[ComplexSpec]:
    """The specs that give a sequence and no structure, so need folding first."""
    for spec in specs:
        if spec.protein_path is None and spec.protein_sequence:
            yield spec


def fold_sequence_receptors(
    specs: list[ComplexSpec],
    out_dir: str | Path,
    *,
    fold: Callable[[str, Path], Path] | None = None,
) -> dict[str, str]:
    """Fold every sequence-only receptor and return name -> structure path.

    Upstream calls fair-esm's `esmfold_v1` on a GPU. tt-bio folds with its own
    esmfold2 on the card (D-dfd-input-5). The folder is injected so this module
    stays host-side and testable; `fold(sequence, path) -> path` is the
    contract. An already-written structure is reused, as upstream does.
    """
    if fold is None:
        from .folding import fold_with_esmfold2 as fold
    out_dir = Path(out_dir)
    paths: dict[str, str] = {}
    for spec in iter_sequence_receptors(specs):
        target = out_dir / spec.name / f"{spec.name}_esmfold2.cif"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            fold(spec.protein_sequence, target)
        paths[spec.name] = str(target)
    return paths


def chain_sequences_for_embedding(path: str | Path) -> list[str]:
    """The per-chain sequences a receptor file contributes, in chain order.

    This is what tt-bio's ESM-2 is asked to embed. Chain order is upstream's
    (`np.unique` over segment+chain), because the embeddings are concatenated
    in it and the receptor nodes are indexed in it.
    """
    sequences, _ = parse_receptor(path).chain_sequences()
    return sequences


def split_embedding_by_chain(embedding: np.ndarray, sequences: list[str]) -> list[np.ndarray]:
    """Split one concatenated per-residue embedding back into per-chain blocks."""
    out, start = [], 0
    for seq in sequences:
        out.append(embedding[start:start + len(seq)])
        start += len(seq)
    if start != len(embedding):
        raise ValueError(
            f"embedding has {len(embedding)} residues, chains sum to {start}")
    return out

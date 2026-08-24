"""X-Cell's input pipeline: expression normalisation, gene subsampling, cell-set construction.

Everything here is specified in the preprint's appendix A.10 and needs no weights, so it is
implemented and tested even though no trained X-Cell checkpoint exists. It earns its own module
because it is pure host data handling with no ttnn in it, and because the AnnData dependency is
optional -- `anndata` is imported lazily so `tt_bio.xcell` works without it and only the `.h5ad`
entry point asks for it, which is what upstream's own stub does too.

The two details that are easy to get wrong and are therefore asserted in
`tests/test_xcell_data.py`: expression is clamped by the *model* and not here (A3 clips at 512
inside the value encoder), and the perturbed gene is FORCED into the subsample at position 1,
right after CLS. A subsample that drops the knockdown target asks the model to predict the effect
of a perturbation it cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CP10K_TARGET = 1e4
MAX_GENES = 4000        # G' <= 4000 (A.10.2)
SET_SIZE = 64           # S, cells per set (A.10.1)


def _require_anndata():
    try:
        import anndata  # noqa: F401
    except ImportError as exc:                                # pragma: no cover - env dependent
        raise ImportError(
            "reading .h5ad needs anndata, which tt-bio does not install by default. "
            "`pip install anndata`, or pass expression as a numpy array instead."
        ) from exc
    return anndata


def normalize(x: np.ndarray, *, pre_normalized: bool = False) -> np.ndarray:
    """A.10.2: raw counts -> CP10k -> log1p. A pre-normalised matrix passes through.

    `pre_normalized` exists because the appendix names a dataset it applies to (Replogle-Nadig
    ships already CP10k+log1p), and normalising twice is silent: the result is still finite, still
    the right shape, and wrong.
    """
    x = np.asarray(x, dtype=np.float32)
    if pre_normalized:
        return x
    total = x.sum(axis=1, keepdims=True)
    total = np.where(total > 0, total, 1.0)          # an all-zero cell stays all-zero
    return np.log1p(CP10K_TARGET * x / total)


def subsample_genes(n_genes_total: int, pert_index: int | None, n_genes: int,
                    rng: np.random.Generator) -> np.ndarray:
    """A.10.2: random subsample to `n_genes`, with the perturbed gene forced to position 0.

    Position 0 of the returned gene order becomes position 1 of the token sequence, because the
    CLS token occupies position 0. Returns indices into the full gene axis.
    """
    if n_genes < 1:
        raise ValueError(f"n_genes must be >= 1, got {n_genes}")
    n_genes = min(n_genes, n_genes_total)
    if pert_index is None:
        return rng.choice(n_genes_total, size=n_genes, replace=False)
    if not 0 <= pert_index < n_genes_total:
        raise ValueError(f"pert_index {pert_index} outside 0..{n_genes_total - 1}")
    rest = np.delete(np.arange(n_genes_total), pert_index)
    picked = rng.choice(rest, size=n_genes - 1, replace=False)
    return np.concatenate([[pert_index], picked])


def build_sets(n_cells: int, set_size: int = SET_SIZE,
               rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """A.10.1: permute the cells and chunk into sets of `set_size`.

    "Incomplete sets are padded by sampling with replacement", so every returned set has exactly
    `set_size` members and the last one may repeat cells. That is upstream's construction, not a
    convenience: a short final set would change the set-level statistics the model was trained on.
    """
    if n_cells < 1:
        raise ValueError(f"need at least one cell, got {n_cells}")
    if set_size < 1:
        raise ValueError(f"set_size must be >= 1, got {set_size}")
    rng = rng if rng is not None else np.random.default_rng(0)
    order = rng.permutation(n_cells)
    sets = []
    for start in range(0, n_cells, set_size):
        chunk = order[start:start + set_size]
        if len(chunk) < set_size:
            extra = rng.choice(order, size=set_size - len(chunk), replace=True)
            chunk = np.concatenate([chunk, extra])
        sets.append(chunk)
    return sets


@dataclass(frozen=True)
class PerturbInput:
    """One ready-to-run cell set: what the model's forward pass takes, before tokenisation."""

    values: np.ndarray          # [S, G] log1p CP10k control expression
    gene_indices: np.ndarray    # [G] indices into the source gene axis, perturbed gene first
    gene_names: list[str]       # [G] the same genes by name, for writing the output back out
    pert_gene: str


def prepare(expression: np.ndarray, gene_names: list[str], pert_gene: str, *,
            n_genes: int = MAX_GENES, set_size: int = SET_SIZE,
            pre_normalized: bool = False,
            rng: np.random.Generator | None = None) -> list[PerturbInput]:
    """Turn a control cell x gene matrix into the cell sets one `predict` call consumes.

    `pert_gene` not being in `gene_names` is an error rather than a warning: the appendix forces
    the perturbed gene into the subsample, so a knockdown target the matrix does not carry cannot
    be represented at all.
    """
    if expression.ndim != 2:
        raise ValueError(f"expression must be 2-D [cells, genes], got {expression.shape}")
    n_cells, n_g = expression.shape
    if len(gene_names) != n_g:
        raise ValueError(f"{len(gene_names)} gene names for {n_g} expression columns")
    try:
        pert_index = gene_names.index(pert_gene)
    except ValueError:
        raise ValueError(
            f"perturbed gene {pert_gene!r} is not in this dataset's {n_g} genes. X-Cell forces "
            "the knockdown target into the gene subsample, so it must be present."
        ) from None

    rng = rng if rng is not None else np.random.default_rng(0)
    values = normalize(expression, pre_normalized=pre_normalized)
    picked = subsample_genes(n_g, pert_index, n_genes, rng)
    names = [gene_names[i] for i in picked]
    return [
        PerturbInput(values=values[np.ix_(cells, picked)], gene_indices=picked,
                     gene_names=names, pert_gene=pert_gene)
        for cells in build_sets(n_cells, set_size, rng)
    ]


def read_h5ad(paths) -> tuple[np.ndarray, list[str]]:
    """Read one or more `.h5ad` files into (expression, gene_names), pooling cells across files."""
    ad = _require_anndata()
    if isinstance(paths, (str, bytes)) or hasattr(paths, "__fspath__"):
        paths = [paths]
    adatas = [ad.read_h5ad(p) for p in paths]
    merged = adatas[0] if len(adatas) == 1 else ad.concat(adatas, merge="same")
    x = merged.X
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    return np.asarray(x, dtype=np.float32), list(merged.var_names)

"""The oracle seam RFD4's reward-guided beam search scores candidates through.

Beam search branches a trajectory, scores every candidate with a structure-prediction oracle and
keeps the best; the reward is a weighted sum of named metric outputs, and a candidate whose oracle
failed scores -inf and is dropped. Upstream's oracle is AlphaFold3 in a container, or OpenFold3
weights through the same code path (`configs/metrics/refolding_oracles/{af3,of3}.yaml`); its reward
keys are AF3's vocabulary: `plddt_coi`, `min_ipae_coi`, `overall_plddt` and the rest.

tt-bio's oracles return their own dicts: a full (N, N) PAE in Angstrom, per-atom pLDDT, pTM, ipTM
and a chain-pair ipTM matrix, but no per-chain pLDDT, no chain-pair PAE and no chain-of-interest
reduction. This module is the one place that bridges the two, in two steps:

1. `summary_confidences` turns token-level arrays into the AF3-shaped summary dict, computing the
   per-chain and chain-pair reductions AF3 reports and tt-bio does not.
2. `extract_confidence_metrics` translates that dict into the flat reward keys, a port of
   `rfproteina/metrics/oracles/oracle_metric.py::extract_confidence_metrics` kept key-for-key and
   scale-for-scale so upstream's `RewardModel` consumes the output unchanged.

Scales follow AF3: pLDDT, pTM and ipTM arrive on 0..1 and leave on 0..100; PAE is Angstrom
throughout. A backend that reports pLDDT on 0..100 (several tt-bio models do) divides before
calling in, which is the adapter's job and not this module's.

`interface_scores.py` also reduces a PAE matrix per chain pair, for ipSAE/pDockQ. Those are
different quantities under a cutoff; the means and minima here are AF3's, and the two are not
interchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

#: Reward keys read straight off the summary dict, as (source, output, scale).
_GLOBAL_METRICS = [
    ("overall_plddt", "overall_plddt", 100.0),
    ("overall_pae", "overall_pae", 1.0),
    ("ptm", "overall_ptm", 100.0),
    ("iptm", "overall_iptm", 100.0),
    ("ranking_score", "ranking_score", 1.0),
    ("fraction_disordered", "fraction_disordered", 1.0),
    ("has_clash", "has_clash", 1.0),
]

#: Per-chain values taken at the chain of interest.
_COI_CHAIN_METRICS = [
    ("chain_plddt", "plddt_coi", 100.0),
    ("chain_ptm", "ptm_coi", 100.0),
    ("chain_pae", "pae_coi", 1.0),
]

#: Reductions over the COI's chain pairs, as (source, output, reduce, scale).
_COI_INTERFACE_METRICS = [
    ("chain_pair_pae", "ipae_coi", np.mean, 1.0),
    ("chain_pair_pae_min", "min_ipae_coi", np.min, 1.0),
    ("chain_pair_iptm", "iptm_coi", np.mean, 100.0),
]

#: Emitted for every chain pair as `<prefix>_<chain_i>_<chain_j>`, forward and reverse averaged.
_CHAIN_PAIR_METRICS = [
    ("chain_pair_pae_min", "min_ipae", 1.0),
    ("chain_pair_iptm", "iptm", 100.0),
]


def summary_confidences(
    *,
    plddt: Sequence[float],
    pae: np.ndarray,
    chains: Sequence,
    ptm: float,
    iptm: float | None = None,
    chain_pair_iptm: np.ndarray | None = None,
    chain_ptm: Sequence[float] | None = None,
    ranking_score: float | None = None,
    fraction_disordered: float | None = None,
    has_clash: float | None = None,
) -> tuple[dict, list]:
    """Build the AF3-shaped summary-confidences dict from one fold's token-level outputs.

    `plddt` is (N,) on 0..1, `pae` (N, N) in Angstrom, `chains` (N,) chain labels per token. The
    chain axis of `chain_pair_iptm` and `chain_ptm`, when given, must follow first appearance of
    each label in `chains`, which is the order returned alongside the dict.

    Per-chain pLDDT is the mean over that chain's tokens; `chain_pae[i]` the mean of the diagonal
    block; `chain_pair_pae[i][j]` the mean of the i-rows x j-columns block and
    `chain_pair_pae_min[i][j]` its minimum. PAE is directional (the row is the aligned residue),
    so the two off-diagonal blocks differ and are kept apart here; the callers that want one
    number per unordered pair average them.
    """
    plddt = np.asarray(plddt, dtype=float)
    pae = np.asarray(pae, dtype=float)
    chains = np.asarray(chains)
    n = len(chains)
    if plddt.shape != (n,) or pae.shape != (n, n):
        raise ValueError(f"shape mismatch: {n} tokens, plddt {plddt.shape}, pae {pae.shape}")

    order = list(dict.fromkeys(chains.tolist()))
    masks = [chains == label for label in order]
    k = len(order)

    chain_pair_pae = np.empty((k, k), dtype=float)
    chain_pair_pae_min = np.empty((k, k), dtype=float)
    for i, mi in enumerate(masks):
        for j, mj in enumerate(masks):
            block = pae[np.ix_(mi, mj)]
            chain_pair_pae[i, j] = block.mean()
            chain_pair_pae_min[i, j] = block.min()

    summary = {
        "overall_plddt": float(plddt.mean()),
        "overall_pae": float(pae.mean()),
        "ptm": float(ptm),
        "chain_plddt": [float(plddt[m].mean()) for m in masks],
        "chain_pae": [float(chain_pair_pae[i, i]) for i in range(k)],
        "chain_pair_pae": chain_pair_pae.tolist(),
        "chain_pair_pae_min": chain_pair_pae_min.tolist(),
    }
    if iptm is not None:
        summary["iptm"] = float(iptm)
    if chain_ptm is not None:
        if len(chain_ptm) != k:
            raise ValueError(f"chain_ptm has {len(chain_ptm)} entries for {k} chains {order}")
        summary["chain_ptm"] = [float(v) for v in chain_ptm]
    if chain_pair_iptm is not None:
        chain_pair_iptm = np.asarray(chain_pair_iptm, dtype=float)
        if chain_pair_iptm.shape != (k, k):
            raise ValueError(f"chain_pair_iptm is {chain_pair_iptm.shape} for {k} chains {order}")
        summary["chain_pair_iptm"] = chain_pair_iptm.tolist()
    for key, value in (("ranking_score", ranking_score),
                       ("fraction_disordered", fraction_disordered),
                       ("has_clash", has_clash)):
        if value is not None:
            summary[key] = float(value)
    return summary, order


def _coi_interface_vals(matrix, coi_idx: int, n_chains: int) -> list:
    """The COI's interface values against every other chain, forward and reverse averaged."""
    vals = []
    for j in range(n_chains):
        if j == coi_idx:
            continue
        pair = [v for v in (matrix[coi_idx][j], matrix[j][coi_idx]) if v is not None]
        if pair:
            vals.append(sum(pair) / len(pair))
    return vals


def extract_confidence_metrics(summary: dict, chain_of_interest, chains_in_order: list) -> dict:
    """Flatten an AF3-shaped summary dict into the reward keys beam search weights.

    Port of upstream's helper of the same name. Besides the COI reductions it emits one
    `min_ipae_<i>_<j>` and `iptm_<i>_<j>` per chain pair, which `min_ipae_coi` cannot express:
    that key is a single minimum over the COI's pairs, so the tightest interface hides the rest.
    """
    metrics: dict[str, float] = {}

    for src, dst, scale in _GLOBAL_METRICS:
        if (v := summary.get(src)) is not None:
            metrics[dst] = float(v) * scale

    for src, prefix, scale in _CHAIN_PAIR_METRICS:
        matrix = summary.get(src)
        if matrix is None:
            continue
        for i, j in combinations(range(len(chains_in_order)), 2):
            if chains_in_order[i] is None or chains_in_order[j] is None:
                continue
            vals = [v for v in (matrix[i][j], matrix[j][i]) if v is not None]
            if vals:
                metrics[f"{prefix}_{chains_in_order[i]}_{chains_in_order[j]}"] = float(np.mean(vals)) * scale

    if chain_of_interest is None or chain_of_interest not in chains_in_order:
        return metrics

    coi_idx = list(chains_in_order).index(chain_of_interest)
    n_chains = len(chains_in_order)

    for src, dst, scale in _COI_CHAIN_METRICS:
        v = summary.get(src)
        if v is not None and v[coi_idx] is not None:
            metrics[dst] = float(v[coi_idx]) * scale

    for src, dst, reduce_fn, scale in _COI_INTERFACE_METRICS:
        matrix = summary.get(src)
        if matrix is None:
            continue
        if vals := _coi_interface_vals(matrix, coi_idx, n_chains):
            metrics[dst] = float(reduce_fn(vals)) * scale

    return metrics


@dataclass(frozen=True)
class OracleRequest:
    """One candidate queued for scoring.

    `chain_of_interest` is the label the `*_coi` reward keys are about: the designed chain for a
    binder, the whole monomer for unconditional generation. `out_dir` None means the refold is
    scored and dropped, which is what beam search wants for the candidates it prunes.
    """

    structure: str              # mmCIF or PDB text of the decoded candidate
    example_id: str
    chain_of_interest: str | None = None
    out_dir: Path | None = None


class Oracle(Protocol):
    """What beam search needs from a scoring model, and nothing more.

    `fold_batch` rather than `fold_one` is the primitive: a checkpoint scores `beam_width *
    n_branch` candidates at once, and that is the only shape in which a Tenstorrent backend can
    amortise one device open, one weight load and one trace capture over the whole set. The device
    is opened in `setup` and held; opening it per candidate would dominate the search.

    Returns one flat dict per request, keyed `"<name>.<output>"` in AF3's vocabulary, in request
    order. A failed fold returns an empty dict for that candidate rather than raising: the reward
    model scores a missing weighted key as -inf, which is how upstream drops it from the beam.
    """

    name: str

    def setup(self) -> None: ...

    def fold_batch(self, jobs: list[OracleRequest]) -> list[dict]: ...

    def teardown(self) -> None: ...


def prefix(name: str, metrics: dict) -> dict:
    """Namespace one fold's metrics under the oracle's config key, as the reward keys read."""
    return {f"{name}.{key}": value for key, value in metrics.items()}

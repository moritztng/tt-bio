"""RF3's confidence reductions: logits to pLDDT, PAE/PDE, pTM/ipTM and a ranking score.

The ConfidenceHead emits logits. Every confidence number a user actually reads -- the
B-factor column of the written structure and `summary_confidences.json` -- comes from
the reductions here, transcribed from upstream: `rf3.metrics.metric_utils`,
`rf3.utils.predicted_error`, `rf3.metrics.predicted_error`, `rf3.metrics.clashing_chains`
and `rf3.inference_engines.rf3.compute_ranking_score`.

Two shapes to keep straight. pLDDT logits are per-token but carry a per-atom-slot axis
folded into the channel: [I, NHEAVY * n_bins], ordered (slot, bin). PAE/PDE are
[I, I, n_bins]. The bin schemes are upstream's `confidence_loss.yaml`.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

# 23, not the 36 that `is_real_atom` is sized to: the last 13 slots are hydrogens and
# the head has no channels for them. Upstream carries the same constant with the same
# comment (`predicted_error.py`, "right now that number is too large (36)").
NHEAVY = 23

# (n_bins, max_value) per head, from configs/trainer/loss/losses/confidence_loss.yaml.
BINS = {"plddt": (50, 1.0), "pae": (64, 32.0), "pde": (64, 32.0),
        "exp_resolved": (2, 1.0)}


def bin_midpoints(max_value: float, n_bins: int) -> torch.Tensor:
    """Upstream's `find_bin_midpoints`: NOT a uniform grid.

    The first and last midpoints sit half a bin outside the linspace, so a naive
    `(i + 0.5) * max_value / n_bins` is wrong at both ends.
    """
    bin_size = max_value / n_bins
    bins = torch.linspace(bin_size, max_value - bin_size, n_bins - 1)
    mid = (bins[1:] + bins[:-1]) / 2
    return torch.cat([(bins[0] - bin_size / 2)[None], mid, bins[-1:] + bin_size / 2])


def unbin(logits: torch.Tensor, kind: str) -> torch.Tensor:
    """Expectation of a binned prediction over its bin midpoints. Bins are last."""
    n_bins, max_value = BINS[kind]
    assert logits.shape[-1] == n_bins, (kind, logits.shape)
    p = logits.float().softmax(-1)
    return (p * bin_midpoints(max_value, n_bins)).sum(-1)


def atomwise_plddt(plddt_logits: torch.Tensor,
                   is_real_atom: torch.Tensor) -> torch.Tensor:
    """Per-atom pLDDT in [0, 1], in the featurizer's atom order.

    `plddt_logits` [I, NHEAVY * n_bins], `is_real_atom` [I, >=NHEAVY]. The boolean
    index flattens (token, slot) row-major, which is the atom order the featurizer's
    `atom_array` is in -- so the result lines up with it element for element.
    """
    n_bins = BINS["plddt"][0]
    i = plddt_logits.shape[-2] if plddt_logits.ndim > 1 else 1
    per_slot = unbin(plddt_logits.reshape(i, NHEAVY, n_bins), "plddt")   # [I, NHEAVY]
    return per_slot[is_real_atom[..., :NHEAVY].bool()]


def ptm(pae_logits: torch.Tensor, pairs: torch.Tensor | None = None) -> float:
    """Predicted TM-score from the PAE logits ([I, I, n_bins]).

    `pairs` [I, I] restricts which pairs count; None means all of them (pTM). The
    normalisation is TM-score's own d0, floored at 19 residues as upstream does.
    """
    n_bins, max_value = BINS["pae"]
    n_token = pae_logits.shape[-2]
    if pairs is None:
        pairs = torch.ones((n_token, n_token), dtype=torch.bool)
    d0 = 1.24 * (max(n_token, 19) - 15.0) ** (1 / 3) - 1.8
    weight = 1 / (1 + (bin_midpoints(max_value, n_bins) / d0) ** 2)
    per_pair = (pae_logits.float().softmax(-1) * weight).sum(-1)          # [I, I]
    per_row = (per_pair * pairs).sum(-1) / (pairs.sum(-1) + 1e-6)
    return float(per_row.max())


def iptm(pae_logits: torch.Tensor, asym_id: torch.Tensor) -> float | None:
    """Interface pTM: pTM restricted to cross-chain pairs. None for a single chain."""
    asym = asym_id.reshape(-1)
    cross = asym[None, :] != asym[:, None]
    return ptm(pae_logits, cross) if bool(cross.any()) else None


def has_clash(atom_array, coord: np.ndarray) -> bool:
    """Upstream's clash test between every pair of polymer pn-units: a pair clashes
    when more than 100 atom pairs are within 1.1 A, or more than half the larger
    chain's atom count is."""
    if not {"pn_unit_id", "is_polymer"} <= set(atom_array.get_annotation_categories()):
        return False
    pn = np.asarray(atom_array.pn_unit_id)
    poly = np.asarray(atom_array.is_polymer, dtype=bool)
    for a, b in itertools.combinations(sorted(set(pn.tolist())), 2):
        ia, ib = pn == a, pn == b
        if not (poly[ia].any() and poly[ib].any()):
            continue
        d = torch.cdist(torch.from_numpy(coord[ia]).float(),
                        torch.from_numpy(coord[ib]).float())
        n = int((d < 1.1).sum())
        if n > 100 or n / (max(int(ia.sum()), int(ib.sum())) + 1e-6) > 0.5:
            return True
    return False


def ranking_score(iptm_v: float | None, ptm_v: float | None,
                  clash: bool) -> float:
    """0.8 * ipTM + 0.2 * pTM - 100 * has_clash; a monomer scores on pTM alone."""
    if iptm_v is None:
        iptm_v = ptm_v if ptm_v is not None else 0.0
    return 0.8 * iptm_v + 0.2 * (ptm_v or 0.0) - 100 * int(clash)


def _chain_masks(labels):
    """Per-chain 1D masks and per-chain-pair 2D masks, upstream's construction.

    The pair masks are SYMMETRIC (both off-diagonal blocks). That matters for PAE,
    which is not a symmetric matrix -- taking one block would report a different
    number than the reference does.
    """
    labels = np.asarray(labels)
    chains = list(np.unique(labels))
    ones = {c: torch.from_numpy(labels == c) for c in chains}
    pairs = {}
    for a, b in itertools.combinations(chains, 2):
        pairs[(a, b)] = (torch.outer(ones[a], ones[b])
                         | torch.outer(ones[b], ones[a]))
    return chains, ones, pairs


def _masked_mean(m: torch.Tensor, mask: torch.Tensor) -> float:
    return float((m * mask).sum() / (mask.sum() + 1e-6))


def summary(out: dict, f: dict, is_real_atom: torch.Tensor,
            chain_iid_token_lvl, atom_array, coord: np.ndarray) -> dict:
    """The AF3-style `summary_confidences` dict, in upstream's key order and rounding.

    `out` is `RF3.predict`'s return. Chains are grouped by `chain_iid_token_lvl` (the
    featurizer's per-token chain label), which is what upstream groups by -- `asym_id`
    is what pTM/ipTM use, and the two are not interchangeable on an input with
    multiple entities per chain.

    One upstream quirk reproduced deliberately: the `chain_ptm` key holds the per-chain
    mean pLDDT, not a per-chain pTM (`predicted_error.py:380` reads
    `chain_plddt.get(c)`). Fixing the name here would make this JSON disagree with the
    reference's, so it stays.
    """
    plddt_slots = unbin(
        out["plddt_logits"].reshape(-1, NHEAVY, BINS["plddt"][0]), "plddt")
    real = is_real_atom[..., :NHEAVY].bool()
    plddt = plddt_slots[real]
    pae_l = out["pae_logits"].reshape(*out["pae_logits"].shape[-3:])
    pde_l = out["pde_logits"].reshape(*out["pde_logits"].shape[-3:])
    pae, pde = unbin(pae_l, "pae"), unbin(pde_l, "pde")

    chains, ones, pairs = _chain_masks(chain_iid_token_lvl)
    n = len(chains)

    def matrix(m, reduce):
        out_m = [[None] * n for _ in range(n)]
        for i, a in enumerate(chains):
            for j, b in enumerate(chains):
                if i == j or (a, b) not in pairs:
                    continue
                out_m[i][j] = round(reduce(m, pairs[(a, b)]), 2)
        return out_m

    def masked_min(m, mask):
        return float(m[mask].min()) if bool(mask.any()) else 0.0

    p = ptm(pae_l)
    ip = iptm(pae_l, f["asym_id"])
    clash = has_clash(atom_array, coord)
    return {
        "chain_ptm": [round(_masked_mean(plddt_slots, real & ones[c][:, None]), 2)
                      for c in chains],
        "chain_pair_pae_min": matrix(pae, masked_min),
        "chain_pair_pde_min": matrix(pde, masked_min),
        "chain_pair_pae": matrix(pae, _masked_mean),
        "chain_pair_pde": matrix(pde, _masked_mean),
        "overall_plddt": round(float(plddt.mean()), 4),
        "overall_pde": round(float(pde.mean()), 4),
        "overall_pae": round(float(pae.mean()), 4),
        "ptm": p,
        "iptm": ip,
        "has_clash": clash,
        "ranking_score": round(ranking_score(ip, p, clash), 4),
    }

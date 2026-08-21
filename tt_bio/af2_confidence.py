"""The five confidence scalars PXDesign filters designs on: pLDDT, pTM, ipTM, pAE, ipAE.

Read from ColabDesign's `af/loss.py:188-259`, `alphafold/common/confidence.py:53-169` and
`af/design.py:112-114`. Two details are easy to get wrong and both change the number:

* `pae` and `i_pae` are computed on the **symmetrised** matrix, `(p + p.T) / 2`, before any mask
  is applied, and `pae` is binder rows against *all* columns, not binder against binder.
* the binder is the **last** `binder_len` residues and the target the first ones
  (`loss.py:38`).

PXDesign persists these rounded to two decimals (`main_af2_complex.py:80-88`) and applies its
`af2_easy` / `af2_opt` thresholds to the rounded values, so the filter's real sensitivity is
5e-3. No model import here: the ttnn arm consumes this file too.
"""

from __future__ import annotations

import torch

# `predicted_aligned_error.max_error_bin`; `get_pae_loss` divides by it to land the loss in [0,1].
PAE_MAX_ERROR_BIN = 31.0


def _mask_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    """`loss.mask_loss`, epsilon and all."""
    return float((value * mask).sum() / (1e-8 + mask.sum()))


def _bin_centers(breaks: torch.Tensor) -> torch.Tensor:
    """`confidence._calculate_bin_centers`: half a step up, plus one catch-all centre."""
    breaks = breaks.float()
    step = breaks[1] - breaks[0]
    centers = breaks + step / 2
    return torch.cat([centers, centers[-1:] + step])


def plddt_per_residue(logits: torch.Tensor) -> torch.Tensor:
    """`loss.get_plddt`: 50 bins of width 1/50, centred at 0.01, 0.03, ..., 0.99."""
    num_bins = logits.shape[-1]
    width = 1.0 / num_bins
    centers = torch.arange(num_bins, dtype=torch.float32) * width + 0.5 * width
    return (torch.softmax(logits.float(), -1) * centers).sum(-1)


def expected_aligned_error(logits: torch.Tensor, breaks: torch.Tensor) -> torch.Tensor:
    """`loss.get_pae`: the expected error in angstroms, per ordered residue pair."""
    return (torch.softmax(logits.float(), -1) * _bin_centers(breaks)).sum(-1)


def predicted_tm_score(logits: torch.Tensor, breaks: torch.Tensor,
                       residue_weights: torch.Tensor,
                       asym_id: torch.Tensor | None = None) -> float:
    """`confidence.predicted_tm_score`. With `asym_id` this is ipTM: only cross-chain pairs."""
    centers = _bin_centers(breaks)
    d0 = 1.24 * (residue_weights.sum().clamp(min=19.0) - 15) ** (1 / 3) - 1.8
    per_bin = 1.0 / (1 + centers.square() / d0.square())
    per_pair = (torch.softmax(logits.float(), -1) * per_bin).sum(-1)
    pair_mask = (torch.ones_like(per_pair) if asym_id is None
                 else (asym_id[:, None] != asym_id[None, :]).float())
    per_pair = per_pair * pair_mask
    pair_weights = pair_mask * residue_weights[None, :] * residue_weights[:, None]
    normed = pair_weights / (1e-8 + pair_weights.sum(-1, keepdim=True))
    return float(((per_pair * normed).sum(-1) * residue_weights).max())


def confidence_scalars(plddt_logits: torch.Tensor, pae_logits: torch.Tensor,
                       pae_breaks: torch.Tensor, seq_mask: torch.Tensor,
                       asym_id: torch.Tensor, *,
                       binder_len: int | None = None) -> dict[str, float]:
    """The scalars ColabDesign logs for one design, from the last recycle's outputs.

    `binder_len` selects the protocol: the binder one scores pLDDT and pAE over the binder and
    adds the interface pAE (`loss.py:34-57`), and `binder_len=None` is hallucination, which
    scores every residue and keeps pAE inside a chain (`loss.py:149-173`).
    """
    seq_mask = seq_mask.float()
    plddt = plddt_per_residue(plddt_logits)
    pae = expected_aligned_error(pae_logits, pae_breaks) / PAE_MAX_ERROR_BIN
    pae = (pae + pae.T) / 2

    if binder_len is None:
        mask_1d = seq_mask
        same_chain = (asym_id[:, None] == asym_id[None, :]).float()
        pae_mask = seq_mask[:, None] * seq_mask[None, :] * same_chain
        target = None
    else:
        mask_1d = torch.zeros_like(seq_mask)
        mask_1d[-binder_len:] = seq_mask[-binder_len:]
        target = torch.zeros_like(seq_mask)
        target[:-binder_len] = seq_mask[:-binder_len]
        pae_mask = mask_1d[:, None].expand_as(pae)

    out = {
        "plddt": 1.0 - _mask_mean(1.0 - plddt, mask_1d),
        "ptm": predicted_tm_score(pae_logits, pae_breaks, seq_mask),
        "i_ptm": predicted_tm_score(pae_logits, pae_breaks, seq_mask, asym_id),
        "pae": _mask_mean(pae, pae_mask),
    }
    if target is not None:
        out["i_pae"] = _mask_mean(pae, mask_1d[:, None] * target[None, :])
        out["unscaled_i_pae"] = out["i_pae"] * PAE_MAX_ERROR_BIN
    return out

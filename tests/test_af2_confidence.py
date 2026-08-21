"""Card-free tests for the five filter scalars.

The reference values live in `scripts/af2_port/parity_artifacts/*/ref_taps*.npz` and are scored
by `scripts/af2_port/tap_gate.py`, which needs the 373 MB checkpoint. What is testable here is
the reduction itself: the bin centres, the symmetrisation, which residues each mask selects.
Each of these was verified to fail with the behaviour removed.
"""
from __future__ import annotations

import torch

from tt_bio import af2_confidence as conf


def _one_hot_logits(shape, index):
    logits = torch.full(shape, -30.0)
    logits[..., index] = 30.0
    return logits


def test_plddt_bin_centres_run_from_0_01_to_0_99():
    assert abs(float(conf.plddt_per_residue(_one_hot_logits((1, 50), 0))) - 0.01) < 1e-6
    assert abs(float(conf.plddt_per_residue(_one_hot_logits((1, 50), 49))) - 0.99) < 1e-6


def test_pae_adds_a_catch_all_bin_above_the_last_break():
    breaks = torch.linspace(0.0, 31.0, 63)
    step = float(breaks[1] - breaks[0])
    centres = conf._bin_centers(breaks)
    assert len(centres) == 64
    assert abs(float(centres[0]) - step / 2) < 1e-6
    assert abs(float(centres[-2]) - (31.0 + step / 2)) < 1e-5
    assert abs(float(centres[-1]) - (31.0 + 3 * step / 2)) < 1e-5


def test_pae_is_symmetrised_before_masking():
    """`get_pae_loss` averages the matrix with its transpose, so the two triangles must agree."""
    breaks = torch.linspace(0.0, 31.0, 63)
    logits = torch.full((4, 4, 64), -30.0)
    logits[..., 0] = 30.0
    logits[0, 3, 0], logits[0, 3, 63] = -30.0, 30.0        # one loud off-diagonal pair
    seq_mask, asym_id = torch.ones(4), torch.tensor([0, 0, 1, 1])
    lower = conf.confidence_scalars(torch.zeros(4, 50), logits, breaks, seq_mask, asym_id,
                                    binder_len=2)["i_pae"]
    logits_t = logits.transpose(0, 1).contiguous()
    upper = conf.confidence_scalars(torch.zeros(4, 50), logits_t, breaks, seq_mask, asym_id,
                                    binder_len=2)["i_pae"]
    assert abs(lower - upper) < 1e-6


def test_the_binder_is_the_last_residues():
    """pLDDT is averaged over the binder only, so a bad target must not move it."""
    breaks = torch.linspace(0.0, 31.0, 63)
    logits = _one_hot_logits((6, 50), 49)
    logits[:4] = _one_hot_logits((4, 50), 0)               # a hopeless target
    got = conf.confidence_scalars(logits, torch.zeros(6, 6, 64), breaks, torch.ones(6),
                                  torch.tensor([0, 0, 0, 0, 1, 1]), binder_len=2)
    assert abs(got["plddt"] - 0.99) < 1e-5


def test_hallucination_scores_every_residue():
    breaks = torch.linspace(0.0, 31.0, 63)
    logits = _one_hot_logits((6, 50), 49)
    logits[:4] = _one_hot_logits((4, 50), 0)
    got = conf.confidence_scalars(logits, torch.zeros(6, 6, 64), breaks, torch.ones(6),
                                  torch.zeros(6, dtype=torch.long))
    assert abs(got["plddt"] - (4 * 0.01 + 2 * 0.99) / 6) < 1e-5
    assert "i_pae" not in got


def test_iptm_is_zero_for_a_single_chain():
    """No cross-chain pair means an all-zero pair mask, and the reference returns 0, not NaN."""
    breaks = torch.linspace(0.0, 31.0, 63)
    logits = _one_hot_logits((200, 200, 64), 0)
    weights = torch.ones(200)
    assert conf.predicted_tm_score(logits, breaks, weights, torch.zeros(200)) == 0.0
    # d0 grows with chain length, so a confident 200-residue prediction scores near 1.
    assert conf.predicted_tm_score(logits, breaks, weights) > 0.99


def test_unscaled_interface_pae_is_the_scaled_one_times_the_max_error_bin():
    breaks = torch.linspace(0.0, 31.0, 63)
    got = conf.confidence_scalars(torch.zeros(4, 50), torch.randn(4, 4, 64), breaks,
                                  torch.ones(4), torch.tensor([0, 0, 1, 1]), binder_len=2)
    assert abs(got["unscaled_i_pae"] - got["i_pae"] * conf.PAE_MAX_ERROR_BIN) < 1e-9

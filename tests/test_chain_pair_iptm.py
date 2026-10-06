"""ConfidenceHead._chain_confidence must surface the full per-chain-pair ipTM matrix.

The cross-chain ipTM of every chain pair is already computed to derive the per-chain
``chain_iptm`` reduction; this pins that the matrix (``pair_chains_iptm``, matching
boltz2's key) is returned nested {chain_i: {chain_j: ipTM}}, is symmetric, carries the
diagonal boltz2's writer reads as ``chains_ptm``, and that ``chain_iptm[c]`` is exactly
the mean over that chain's cross-chain entries. Host-only -- no device, no network (the
method is pure torch on the PAE logits).
"""
from __future__ import annotations

import json

import torch

from tt_bio.protenix import ConfidenceHead


def _synthetic(chain_sizes: list[int], nb: int = 64, seed: int = 0):
    torch.manual_seed(seed)
    asym = torch.tensor([c for c, n in enumerate(chain_sizes) for _ in range(n)])
    n = asym.numel()
    return torch.randn(n, n, nb), asym


def test_pair_chains_iptm_matrix_and_reduction():
    ids = [0, 1, 2]
    pae_logits, asym = _synthetic([20, 15, 10])
    out = ConfidenceHead._chain_confidence(pae_logits, asym)
    pair, chain_ptm, chain_iptm = (out["pair_chains_iptm"], out["chain_ptm"],
                                   out["chain_iptm"])

    # Full square matrix including self-pairs -- boltz2's shape, whose writer reads
    # pci[i][i] (tt_bio.main._pair_chains). A matrix missing the diagonal looks
    # compatible and then KeyErrors on the consumer.
    assert set(pair) == set(ids)
    for c in ids:
        assert set(pair[c]) == set(ids)
    # symmetric, in range
    for i in ids:
        for j in ids:
            assert pair[i][j] == pair[j][i]
            assert 0.0 <= pair[i][j] <= 1.0
    # the diagonal is that chain's own pTM, not an ipTM
    for k, c in enumerate(ids):
        assert pair[c][c] == chain_ptm[k]
    # chain_iptm[c] is the mean over c's CROSS-chain entries (i.e. the matrix is the
    # source); the diagonal is excluded because it is pTM
    for k, c in enumerate(ids):
        cross = [pair[c][j] for j in ids if j != c]
        assert abs(chain_iptm[k] - round(sum(cross) / len(cross), 6)) < 1e-5


def test_pair_chains_iptm_survives_results_json():
    """results.json is the user-visible surface, so pin what a consumer actually reads:
    the integer chain ids come back as strings, and chains_ptm is the diagonal."""
    ids = [0, 1]
    pae_logits, asym = _synthetic([30, 20])
    pair = ConfidenceHead._chain_confidence(pae_logits, asym)["pair_chains_iptm"]
    row = {"pair_chains_iptm": pair, "chains_ptm": {c: pair[c][c] for c in pair}}
    back = json.loads(json.dumps(row))

    assert sorted(back["pair_chains_iptm"]) == [str(c) for c in ids]
    for i in ids:
        for j in ids:
            assert back["pair_chains_iptm"][str(i)][str(j)] == pair[i][j]
    for c in ids:
        assert back["chains_ptm"][str(c)] == pair[c][c]


def test_single_chain_and_none_return_no_chain_keys():
    pae_logits, _ = _synthetic([10])
    assert ConfidenceHead._chain_confidence(pae_logits, torch.zeros(10)) == {}
    assert ConfidenceHead._chain_confidence(pae_logits, None) == {}


def test_two_chain_pair_is_each_models_own_iptm():
    """OpenFold3/OpenBind-0 and RF3 report the matrix through the same reduction. On two
    chains the pair covers every token, so its entry must be the model's own global ipTM:
    OF3's `_ptm_iptm`, and RF3's `iptm` on its non-uniform bin midpoints."""
    from tt_bio.rf3 import confidence as rf3

    pae_logits, asym = _synthetic([21, 30])
    _ptm, iptm = ConfidenceHead._ptm_iptm(pae_logits, asym)
    assert abs(ConfidenceHead._chain_confidence(pae_logits, asym)["pair_chains_iptm"][0][1]
               - iptm) < 1e-5
    n_bins, max_a = rf3.BINS["pae"]
    pair = ConfidenceHead._chain_confidence(
        pae_logits, asym, centers=rf3.bin_midpoints(max_a, n_bins))["pair_chains_iptm"]
    assert abs(pair[0][1] - rf3.iptm(pae_logits, asym)) < 1e-5


def test_a_frameless_token_cannot_win_a_row():
    """The frame mask the global ipTM honours applies to the matrix too: with every token of
    chain 0 but one frameless, chain 0's pTM is that one token's row."""
    pae_logits, asym = _synthetic([20, 15])
    frame = torch.ones(35, dtype=torch.bool)
    frame[1:20] = False
    masked = ConfidenceHead._chain_confidence(pae_logits, asym, has_frame=frame)
    full = ConfidenceHead._chain_confidence(pae_logits, asym)
    assert masked["pair_chains_iptm"][0][0] <= full["pair_chains_iptm"][0][0]
    assert masked["pair_chains_iptm"][1][1] == full["pair_chains_iptm"][1][1]
    only = ConfidenceHead._chain_confidence(pae_logits[:20, :20], torch.zeros(20))
    assert only == {}
    _p, iptm_masked = ConfidenceHead._ptm_iptm(pae_logits, asym, has_frame=frame)
    assert abs(masked["pair_chains_iptm"][0][1] - iptm_masked) < 1e-5


def test_rows_carry_the_matrix_and_its_diagonal():
    from tt_bio.worker import _chain_rows

    pae_logits, asym = _synthetic([10, 12])
    c = ConfidenceHead._chain_confidence(pae_logits, asym)
    row = _chain_rows(c)
    assert row["pair_chains_iptm"] == c["pair_chains_iptm"]
    assert row["chains_ptm"] == {0: c["pair_chains_iptm"][0][0], 1: c["pair_chains_iptm"][1][1]}
    assert _chain_rows({}) == {}


def test_keys_are_chain_positions_whatever_the_asym_ids():
    """OpenFold3 numbers chains from 1; the matrix is keyed 0, 1, ... like every other model's,
    and the numbers are the same as with 0-based ids."""
    pae_logits, asym = _synthetic([30, 20])
    zero = ConfidenceHead._chain_confidence(pae_logits, asym)
    one = ConfidenceHead._chain_confidence(pae_logits, asym + 1)
    assert sorted(one["pair_chains_iptm"]) == [0, 1]
    assert one == zero

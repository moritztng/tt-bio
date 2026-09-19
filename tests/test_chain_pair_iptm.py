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

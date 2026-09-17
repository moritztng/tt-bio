"""ConfidenceHead._chain_ptm_iptm must surface the full per-chain-pair ipTM matrix.

The cross-chain ipTM of every chain pair is already computed to derive the per-chain
``chain_iptm`` reduction; this pins that the matrix (``pair_chains_iptm``, matching
boltz2's key) is returned nested {chain_i: {chain_j: ipTM}}, is symmetric, and that
``chain_iptm[c]`` is exactly the mean over that chain's pair entries. Host-only -- no
device, no network (the method is pure torch on the PAE logits).
"""
from __future__ import annotations

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
    chain_ptm, chain_iptm, pair = ConfidenceHead._chain_ptm_iptm(pae_logits, asym)

    assert set(pair) == set(ids)
    for c in ids:
        assert set(pair[c]) == {x for x in ids if x != c}
    # symmetric
    for i in ids:
        for j in pair[i]:
            assert pair[i][j] == pair[j][i]
            assert 0.0 <= pair[i][j] <= 1.0
    # chain_iptm[c] is the mean over c's pair entries (i.e. the matrix is the source)
    for k, c in enumerate(ids):
        assert abs(chain_iptm[k] - round(sum(pair[c].values()) / len(pair[c]), 6)) < 1e-5


def test_single_chain_and_none_return_triple_none():
    pae_logits, _ = _synthetic([10])
    assert ConfidenceHead._chain_ptm_iptm(pae_logits, torch.zeros(10)) == (None, None, None)
    assert ConfidenceHead._chain_ptm_iptm(pae_logits, None) == (None, None, None)

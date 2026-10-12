"""pair_to_atom_windows must return exactly what the one-hot einsum it replaced returned."""
from functools import partial

import torch

from tt_bio.boltz2 import get_indexing_matrix, pair_to_atom_windows, single_to_keys


def _einsum_reference(z, atom_to_token, to_keys, K, W):
    B = z.shape[0]
    q = atom_to_token.view(B, K, W, atom_to_token.shape[-1])
    k = to_keys(atom_to_token)
    return torch.einsum("bijd,bwki,bwlj->bwkld", z, q, k)


def test_gather_equals_einsum_with_padding():
    torch.manual_seed(0)
    W, H, n_tok, D = 32, 128, 40, 16
    for B, n_real, n_atoms in ((1, 300, 320), (2, 250, 256)):
        K = n_atoms // W
        tok = torch.sort(torch.randint(0, n_tok, (B, n_real))).values
        a2t = torch.zeros(B, n_atoms, n_tok)
        a2t[:, :n_real].scatter_(-1, tok.unsqueeze(-1), 1.0)
        z = torch.randn(B, n_tok, n_tok, D)
        to_keys = partial(single_to_keys, indexing_matrix=get_indexing_matrix(K, W, H, "cpu"), W=W, H=H)
        ref = _einsum_reference(z, a2t, to_keys, K, W)
        out = pair_to_atom_windows(z, a2t, to_keys, K, W, H)
        assert out.shape == ref.shape
        assert torch.equal(out, ref)

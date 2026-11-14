"""RFD3_ATTN_KDTREE builds the same atom neighbour index as the shipped cdist chain. No device."""
import torch

from tt_bio.rfd3 import model as M


def _case(n, n_seq, k, seed, sigma):
    g = torch.Generator().manual_seed(seed)
    steps = torch.randn((n + 8) // 9, 3, generator=g)
    ca = torch.cumsum(3.8 * steps / steps.norm(dim=1, keepdim=True), 0)
    x = ca.repeat_interleave(9, 0)[:n] + torch.randn(n, 3, generator=g) * (1.2 + sigma)
    mask, seq_idx = M._mask_and_seq_idx(torch.arange(n) // 9, n_seq, k, None, None, n)
    return mask, seq_idx, x.unsqueeze(0).float()


def test_kdtree_index_equals_the_cdist_index():
    # n_seq 2: every row needs a large fill; n_seq 32: most rows are all sequence (no fill)
    for n, n_seq, k in ((700, 2, 64), (700, 32, 64), (130, 4, 128)):
        for seed, sigma in ((0, 0.0), (1, 8.0), (2, 80.0)):
            mask, seq_idx, x = _case(n, n_seq, k, seed, sigma)
            ref = M._neighbours_row_blocked(mask, seq_idx, x, k).sort(-1).values
            assert torch.equal(M._neighbours_kdtree(mask, seq_idx, x, k), ref), (n, n_seq, k, seed)


def test_kdtree_rows_are_k_distinct_sorted_and_keep_the_sequence_entries():
    mask, seq_idx, x = _case(700, 2, 64, 3, 2.0)
    idx = M._neighbours_kdtree(mask, seq_idx, x, 64)[0]
    assert idx.shape == (700, 64) and idx.dtype == torch.long
    for r, row in enumerate(idx):
        assert torch.equal(row, row.sort().values) and len(set(row.tolist())) == 64
        seq = set(seq_idx[r][seq_idx[r] < 700].tolist())
        assert seq <= set(row.tolist())
        assert not mask[r, sorted(set(row.tolist()) - seq)].any()


def test_rows_rebuilt_from_the_previous_step_equal_a_fresh_build():
    # A docking trajectory: the target never moves, so most rows keep last step's neighbours and come from the cache.
    n, k = 900, 64
    mask, seq_idx, x0 = _case(n, 4, k, 5, 0.0)
    g = torch.Generator().manual_seed(6)
    binder = torch.arange(600, n)
    for sigma in (40.0, 10.0, 2.0, 0.5, 0.5):
        x = x0.clone()
        x[0, binder] += torch.randn(len(binder), 3, generator=g) * sigma
        got = M._neighbours_kdtree(mask, seq_idx, x, k)
        M._NN_ROWS_CACHE.clear(); M._STATIC_KNN_CACHE.clear()
        assert torch.equal(got, M._neighbours_kdtree(mask, seq_idx, x, k)), sigma

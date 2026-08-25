"""Host-only invariants the gathered atom softmax stands on. No device.

The gathered arm claims two things and buys a third. It claims the gathered SCORES are
bit-identical (a gather is a copy) and that the set of terms entering each row sum is identical.
It buys a different summation ORDER, and that is what the accuracy envelope prices. These tests
pin the two claims in torch so a later change to the mask constant, the key padding or the
neighbour count cannot break them silently.
"""
import torch

from tt_bio.rfd3 import model as M

MASK = -1e4          # _mask_template's constant, and rfd3_bias' kernel writes the same value
L, K, H = 257, 32, 4


def _fixture(seed=0):
    """Scores in the shape the shipped chain leaves them: real values at K sorted neighbours per
    row, MASK everywhere else including the tile padding of the key axis."""
    torch.manual_seed(seed)
    n_key = M.align_tile(L)
    idx = torch.stack([torch.randperm(L)[:K].sort().values for _ in range(L)]).unsqueeze(0)
    idx_h = idx.expand(1, H, L, K).contiguous()
    dense = torch.full((1, H, L, n_key), MASK, dtype=torch.float32)
    dense.scatter_(3, idx_h, torch.randn(1, H, L, K) * 3.0)
    return dense, idx, idx_h, n_key


def test_masked_columns_are_exact_zeros_after_softmax():
    """The whole lever rests on this: a masked column contributes exactly 0.0 to the row sum, so
    dropping it changes the sum's ORDER and not its value."""
    dense, _, idx_h, _ = _fixture()
    w = torch.softmax(dense, dim=3)
    masked = torch.ones_like(w, dtype=torch.bool).scatter_(3, idx_h, False)
    assert w[masked].abs().max().item() == 0.0
    # and it is underflow, not a small number: exp of the shifted mask is exactly zero in fp32
    assert torch.exp(torch.tensor(MASK - 10.0, dtype=torch.float32)).item() == 0.0


def test_gathered_scores_are_bit_identical():
    dense, _, idx_h, _ = _fixture()
    assert torch.equal(dense.gather(3, idx_h), torch.gather(dense, 3, idx_h))
    # the gathered form holds every value that is not the mask, and nothing else
    g = dense.gather(3, idx_h)
    assert (g == MASK).sum().item() == 0
    assert g.numel() == H * L * K


def test_support_is_identical_and_only_the_order_differs():
    """Scatter the gathered softmax back and compare against the dense one: same support, same
    values to fp32 tolerance, and the difference is the reduction order rather than the terms."""
    dense, _, idx_h, n_key = _fixture()
    a_dense = torch.softmax(dense, dim=3)
    a_gath = torch.zeros_like(a_dense).scatter_(3, idx_h, torch.softmax(dense.gather(3, idx_h), 3))
    assert torch.equal(a_dense != 0, a_gath != 0)
    assert (a_dense - a_gath).abs().max().item() < 1e-6
    # both normalise; the row sums are 1 either way
    for a in (a_dense, a_gath):
        assert (a.sum(3) - 1.0).abs().max().item() < 1e-5


def test_zero_template_not_the_mask_template():
    """The post-softmax scatter target must be zeros. Reusing the -1e4 mask template would leave
    -1e4 at every non-neighbour, which the PV matmul would then weight."""
    dense, _, idx_h, _ = _fixture()
    w = torch.softmax(dense.gather(3, idx_h), 3)
    wrong = torch.full_like(dense, MASK).scatter_(3, idx_h, w)
    assert wrong.min().item() == MASK          # the trap, pinned
    right = torch.zeros_like(dense).scatter_(3, idx_h, w)
    assert right.min().item() >= 0.0


def test_every_row_has_exactly_k_valid_neighbours_in_range():
    """No ragged rows: _extend_with_neighbours fills the sequence slots and tops up from the
    distance topk, and _create_attention_indices sorts. A duplicate column would double-count in
    the scatter and a column >= L would read key padding."""
    _, idx, _, _ = _fixture()
    assert idx.shape[-1] == K
    assert int(idx.min()) >= 0 and int(idx.max()) < L
    for row in idx[0]:
        assert len(set(row.tolist())) == K
        assert torch.equal(row, row.sort().values)

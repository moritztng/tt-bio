"""Host-only invariants for the block-sparse atom attention plan.

The device chain is only correct if the plan is, and the plan is pure host torch, so these run
without a card. The load-bearing one is `test_pos_addresses_the_right_key`: every neighbour's
block-local column has to point at that same neighbour in its block's gathered key rows, or the
attention silently reads someone else's key and still produces a plausible structure.
"""
import torch

from tt_bio.rfd3 import block_sparse as BS

L, K, N_KEY = 6051, 128, 6080
Q = 1216
BUCKETS = (3264, 3488, 4224)


def _index(length=L, n_neigh=K, n_key=N_KEY, seed=0, width=None):
    """A neighbour index with a controllable per-block union width.

    Each row draws its neighbours from a window of `width` keys centred on the row, which is what
    makes the block union narrow; `width=None` draws from the whole key axis, which makes it wide
    and is how the dense-fallback path gets exercised.
    """
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(length):
        if width is None:
            pool = torch.randperm(n_key, generator=g)[:n_neigh]
        else:
            lo = max(0, min(i - width // 2, n_key - width))
            pool = lo + torch.randperm(width, generator=g)[:n_neigh]
        rows.append(torch.sort(pool).values)
    return torch.stack(rows).unsqueeze(0)


def test_plan_fires_on_a_narrow_index():
    plan = BS.plan(_index(width=2000), N_KEY, Q, BUCKETS)
    assert plan is not None
    nb, q_block, u_width, gather, pos = plan
    assert (nb, q_block) == (N_KEY // Q, Q)
    assert u_width in BUCKETS
    assert gather.shape == (nb, u_width)
    assert pos.shape == (nb * q_block, K)


def test_pos_addresses_the_right_key():
    """gather[b, pos[i, k]] is exactly the key indices[0, i, k] asks for."""
    idx = _index(width=2000)
    nb, q_block, u_width, gather, pos = BS.plan(idx, N_KEY, Q, BUCKETS)
    padded = torch.cat([idx[0], idx[0, -1:].expand(nb * q_block - L, K)], 0)
    blocks = torch.arange(nb * q_block) // q_block
    assert torch.equal(gather[blocks.unsqueeze(1), pos], padded)


def test_every_column_is_in_range_and_the_union_is_a_set():
    idx = _index(width=2000)
    nb, q_block, u_width, gather, pos = BS.plan(idx, N_KEY, Q, BUCKETS)
    assert int(pos.min()) >= 0 and int(pos.max()) < u_width
    assert int(gather.min()) >= 0 and int(gather.max()) < N_KEY
    for b in range(nb):
        used = int(pos[b * q_block:(b + 1) * q_block].max()) + 1
        # the occupied prefix of each block's row carries no duplicate key
        assert len(set(gather[b, :used].tolist())) == used


def test_bitmap_matches_the_per_block_sort():
    """The bitmap form is byte-identical to the torch.unique + searchsorted it replaces.

    That equality is the whole licence for using the 8.3x cheaper form: the index is the same
    index, so nothing about the arm's numerics depends on which one built it.
    """
    idx = _index(width=2000)
    nb, q_block, u_width, gather, pos = BS.plan(idx, N_KEY, Q, BUCKETS)
    padded = torch.cat([idx[0], idx[0, -1:].expand(nb * q_block - L, K)], 0)
    g_ref = torch.zeros(nb, u_width, dtype=torch.int64)
    p_ref = torch.zeros(nb * q_block, K, dtype=torch.int64)
    for b in range(nb):
        blk = padded[b * q_block:(b + 1) * q_block]
        u = torch.unique(blk)
        g_ref[b, :u.numel()] = u
        p_ref[b * q_block:(b + 1) * q_block] = torch.searchsorted(u, blk)
    assert torch.equal(gather, g_ref)
    assert torch.equal(pos, p_ref)


def test_narrowest_fitting_bucket_is_chosen():
    idx = _index(width=2000)
    _, _, u_width, _, _ = BS.plan(idx, N_KEY, Q, BUCKETS)
    narrower = [b for b in BUCKETS if b < u_width]
    for b in narrower:
        assert BS.plan(idx, N_KEY, Q, (b,)) is None, "bucket %d should not fit" % b
    assert BS.plan(idx, N_KEY, Q, (u_width,)) is not None


def test_wide_index_falls_back_to_dense():
    """A step whose union exceeds every bucket returns None, which is the dense chain."""
    assert BS.plan(_index(width=None), N_KEY, Q, BUCKETS) is None


def test_batch_above_one_falls_back():
    idx = _index(width=2000)
    assert BS.plan(torch.cat([idx, idx], 0), N_KEY, Q, BUCKETS) is None


def test_misaligned_block_size_falls_back():
    """Q must be a whole number of tile rows: otherwise the reshape re-tiles (E9.1)."""
    idx = _index(width=2000)
    assert Q % 32 == 0
    assert BS.plan(idx, N_KEY, 1520, BUCKETS) is None      # divides 6080, not a multiple of 32
    assert BS.plan(idx, N_KEY, 999, BUCKETS) is None       # neither


def test_block_size_must_divide_the_padded_query_axis():
    idx = _index(width=2000)
    assert N_KEY % 1024 != 0
    assert BS.plan(idx, N_KEY, 1024, BUCKETS) is None


def test_the_gate_fixture_has_no_usable_block_size():
    """The release gate's own fixture cannot run this arm, and no choice of Q rescues it.

    `examples/rfd3_binder.json` is 1350 atoms, so the padded query axis is 1376 = 2**5 * 43 with 43
    prime. Its only multiple-of-32 divisors are 32 and 1376: 43 blocks, whose gathered row count
    nb*U is worse than dense, or one block whose union IS the key axis, which is dense. So the arm
    declines every step there and `release_gate.py --model rfd3` passes with RFD3_BLOCK_SPARSE=1
    having scored the shipped dense chain twice (measured: 0 blocked, 1791 dense-fallback).

    This is pinned rather than fixed because default-OFF is correct given it. If someone changes Q,
    the gate fixture, or plan()'s divisibility policy, this test fails and the coverage claim in
    block_sparse.py's docstring has to be re-stated.
    """
    gate_n_key = 1376                                  # _tile(1350)
    assert gate_n_key % 32 == 0 and gate_n_key % Q != 0
    usable = [q for q in range(32, gate_n_key + 1, 32) if gate_n_key % q == 0]
    assert usable == [32, gate_n_key]

    idx = _index(length=1350, n_key=gate_n_key, width=600)
    assert BS.plan(idx, gate_n_key, Q, BUCKETS) is None
    # Every Q the tile rule allows either splits into 43 blocks or is the whole axis.
    for q in usable:
        planned = BS.plan(idx, gate_n_key, q, BUCKETS)
        nb = gate_n_key // q
        assert planned is None or planned[0] == nb


def test_the_live_atom_counts_are_one_in_thirty_eight():
    """How narrow the arm is, as a number, because five passes measured it on the one fixture
    whose padded axis happens to be 5 * 1216 and nothing said so.

    plan() needs Q to DIVIDE the tile-padded atom axis, which is far stronger than the
    multiple-of-32 rule the design notes state. Pure arithmetic on that condition.
    """
    q_tiles = Q // 32
    assert q_tiles == 38
    live = [a for a in range(256, 12001) if (-(-a // 32) * 32) % Q == 0]
    assert len(live) == 288                            # of 11745 atom counts, 2.45 %
    assert L in live and 1350 not in live              # R4 is live, the gate fixture is not


def test_default_is_off_and_toggles_back():
    was = BS.set_enabled(True)
    try:
        assert BS.enabled() is True
    finally:
        BS.set_enabled(was)
    assert BS.enabled() is was


def test_config_defaults_are_the_measured_optimum():
    q_block, buckets = BS.config()
    assert q_block == 1216
    assert buckets == (3264, 3488, 4224)
    assert q_block % 32 == 0 and N_KEY % q_block == 0
